/**
 * YOLO 辨識引擎服務 (YOLO Inference Service)
 * 
 * 負責 ONNX Runtime 初始化與模型載入。
 * 支援「全域預熱 (Pre-warming)」，讓應用程式啟動時即開始加載大體積權重。
 * 
 * 行動裝置優化版：
 * - 自動偵測行動裝置，限制 WASM 執行緒數量防止過熱
 * - 推理解析度根據裝置自動調整 (320 vs 640)
 */

const isMobile = () => /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

export class YOLOService {
    private session: any = null;
    private isInitializing: boolean = false;
    private isReady: boolean = false;

    // 類別名稱對照表
    public readonly CLASS_NAMES = [
        "apple", "banana", "cabbage", "meat", "orange",
        "rotten apple", "rotten banana", "rotten cabbage",
        "rotten meat", "rotten orange", "rotten spinach", "spinach"
    ];

    // 根據裝置決定推理解析度 (行動裝置用 320 節省算力)
    public get inferenceSize(): number {
        return isMobile() ? 320 : 640;
    }

    /**
     * 預熱模型：在 App 啟動時即刻觸發，不需等待用戶進入 CameraView
     */
    public async prewarm() {
        if (this.isInitializing || this.isReady) return;
        this.isInitializing = true;
        
        const mobile = isMobile();
        console.log(`🧠 [YOLO] 核心啟動中 (背景預熱)... [裝置: ${mobile ? "📱 行動裝置" : "🖥️ 桌面"}]`);
        
        try {
            const ort = (window as any).ort;
            if (!ort) {
                console.warn("⚠️ 找不到 ort 引擎，延後初始化");
                this.isInitializing = false;
                return;
            }

            const baseUrl = import.meta.env.BASE_URL || "/";
            ort.env.wasm.wasmPaths = `${baseUrl}wasm/`;
            
            // ✅ 行動裝置優化：限制執行緒數量為 1，防止過熱 & 卡頓
            // 桌面裝置：最多 4 執行緒發揮多核效能
            const maxThreads = mobile ? 1 : Math.min(navigator.hardwareConcurrency || 4, 4);
            ort.env.wasm.numThreads = maxThreads;
            console.log(`🔧 [YOLO] WASM 執行緒: ${maxThreads} (${mobile ? "省電模式" : "高效模式"})`);

            const modelUrl = `${baseUrl}best.onnx?v=1.0.7`;
            console.log(`📡 [YOLO] 正在載入優化後的模型... (${modelUrl})`);
            
            // ✅ 行動裝置優化：WebGL 比 WASM 更省 CPU（由 GPU 負責矩陣運算）
            // WebGPU 最優，但支援率較低，優先嘗試
            const providers: string[] = [];
            if (!mobile) {
                // 桌面：嘗試 WebGPU → WebGL → WASM
                try {
                    if ('gpu' in navigator) {
                        providers.push("webgpu");
                        console.log("🚀 [YOLO] 啟用 WebGPU 加速");
                    }
                } catch (e) { /* skip */ }
                providers.push("webgl");
            } else {
                // 行動裝置：直接 WebGL → WASM，避免 WebGPU 的高記憶體開銷
                providers.push("webgl");
            }
            providers.push("wasm");

            this.session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: providers,
                graphOptimizationLevel: "basic"
            });

            this.isReady = true;
            this.isInitializing = false;
            console.log(`✅ [YOLO] 核心準備就緒 (Provider: ${this.session.handler?.provider || providers[0]})`);
        } catch (e: any) {
            console.error("❌ [YOLO] 預熱失敗:", e);
            
            // 最後備援：純 WASM + 停用優化
            try {
                console.warn("🔄 [YOLO] 啟動 WASM 極限相容模式...");
                const ort = (window as any).ort;
                const baseUrl = import.meta.env.BASE_URL || "/";
                ort.env.wasm.numThreads = 1; // 確保備援模式用最少執行緒
                this.session = await ort.InferenceSession.create(`${baseUrl}best.onnx?v=1.0.7`, {
                    executionProviders: ["wasm"],
                    graphOptimizationLevel: "disabled"
                });
                this.isReady = true;
                console.log("✅ [YOLO] WASM 相容模式啟動成功");
            } catch (retryError) {
                console.error("💀 [YOLO] 模型最終載入失敗:", retryError);
            }
            this.isInitializing = false;
        }
    }

    public getSession() {
        return this.session;
    }

    public isLoaded() {
        return this.isReady;
    }
}

export const yoloService = new YOLOService();
