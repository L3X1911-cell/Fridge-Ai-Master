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
            ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
            
            // 嘗試啟用 WebGPU (如果瀏覽器支援)
            const providers = ["wasm"];
            try {
                if ('gpu' in navigator) {
                    // 優先使用 WebGPU，對於 FP16 模型支援較好
                    providers.unshift("webgpu");
                    console.log("🚀 [YOLO] 偵測到 WebGPU 支援，已加入執行提供者優先級");
                } else {
                    providers.unshift("webgl");
                }
            } catch (e) {
                providers.unshift("webgl");
            }

            const modelUrl = `${baseUrl}best.onnx?v=1.0.3`;
            
            console.log(`📡 [YOLO] 正在從 ${modelUrl} 載入模型...`);
            
            this.session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: providers,
                // 對於某些量化模型或特定算子 (如 DynamicQuantizeLinear)，
                // 'all' 可能會觸發不相容的優化路徑，改用 'basic' 提高相容性。
                graphOptimizationLevel: "basic" 
            });

            this.isReady = true;
            this.isInitializing = false;
            console.log(`✅ [YOLO] 核心準備就緒 (使用: ${this.session.handler?.provider || "預設"})`);
        } catch (e: any) {
            console.error("❌ [YOLO] 預熱失敗:", e);
            
            // 如果 WebGPU/WebGL 失敗，嘗試最後一搏：純 WASM
            if (e.message?.includes("invalid") || e.message?.includes("fail")) {
                console.warn("⚠️ 嘗試以純 WASM (相容模式) 重新載入...");
                try {
                    const ort = (window as any).ort;
                    const baseUrl = import.meta.env.BASE_URL || "/";
                    this.session = await ort.InferenceSession.create(`${baseUrl}best.onnx?v=1.0.3`, {
                        executionProviders: ["wasm"],
                        graphOptimizationLevel: "disabled"
                    });
                    this.isReady = true;
                    console.log("✅ [YOLO] 使用 WASM 相容模式啟動成功");
                } catch (retryError) {
                    console.error("💀 [YOLO] 最終載入失敗:", retryError);
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
