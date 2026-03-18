/**
 * YOLO 辨識引擎服務 (YOLO Inference Service)
 * 
 * 負責 ONNX Runtime 初始化與模型載入。
 * 支援「全域預熱 (Pre-warming)」，讓應用程式啟動時即開始加載大體積權重。
 */

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

    /**
     * 預熱模型：在 App 啟動時即刻觸發，不需等待用戶進入 CameraView
     */
    public async prewarm() {
        if (this.isInitializing || this.isReady) return;
        this.isInitializing = true;
        
        console.log("🧠 [YOLO] 核心啟動中 (背景預熱)...");
        try {
            const ort = (window as any).ort;
            if (!ort) {
                console.warn("⚠️ 找不到 ort 引擎，延後初始化");
                this.isInitializing = false;
                return;
            }

            const baseUrl = import.meta.env.BASE_URL || "/";
            ort.env.wasm.wasmPaths = `${baseUrl}wasm/`;
            // 啟用多執行緒支援 (SIMD)
            ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);

            const modelUrl = `${baseUrl}best.onnx?v=1.0.7`;
            console.log(`📡 [YOLO] 正在載入優化後的模型... (${modelUrl})`);
            
            // 優先嘗試 WebGPU (效能最高) -> WebGL -> WASM (相容模式)
            const providers = ["wasm"];
            try {
                if ('gpu' in navigator) {
                    providers.unshift("webgpu");
                    console.log("🚀 [YOLO] 啟用 WebGPU 加速");
                } else {
                    providers.unshift("webgl");
                    console.log("🎨 [YOLO] 啟用 WebGL 加速");
                }
            } catch (e) {
                console.warn("⚠️ 硬體加速初始化失敗，回退至 WASM 模式");
            }

            this.session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: providers,
                // 對於量化模型，'basic' 級別比 'all' 更穩定，且載入速度更快（減少圖融合時間）
                graphOptimizationLevel: "basic" 
            });

            this.isReady = true;
            this.isInitializing = false;
            console.log(`✅ [YOLO] 核心準備就緒 (Provider: ${this.session.handler?.provider || "WASM"})`);
        } catch (e: any) {
            console.error("❌ [YOLO] 預熱失敗:", e);
            
            // 如果失敗，嘗試最後一搏：完全禁用優化並強制 WASM
            if (e.message?.includes("invalid") || e.message?.includes("fail")) {
                console.warn("🔄 [YOLO] 偵測到相容性問題，嘗試啟動極限相容模式...");
                try {
                    const ort = (window as any).ort;
                    const baseUrl = import.meta.env.BASE_URL || "/";
                    this.session = await ort.InferenceSession.create(`${baseUrl}best.onnx?v=1.0.6`, {
                        executionProviders: ["wasm"],
                        graphOptimizationLevel: "disabled"
                    });
                    this.isReady = true;
                    console.log("✅ [YOLO] 極限相容模式(WASM-Only) 啟動成功");
                } catch (retryError) {
                    console.error("💀 [YOLO] 模型最終載入失敗:", retryError);
                }
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
