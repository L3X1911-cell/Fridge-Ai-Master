/// <reference types="vite/client" />
import { RefObject, useState, useEffect, useRef } from "react";
import { Camera, Loader2, RefreshCw, Sparkles, Brain, SlidersHorizontal } from "lucide-react";
import { useIngredients } from "../../services/IngredientContext";
import { llmService } from "../../services/llmService";
import { yoloService } from "../../services/yoloService";
import { DetectionSummary } from "../inventory_management/DetectionSummary";
import { notificationService } from "../../services/notificationService";
import { hapticService } from "../../services/hapticService";

// 使用 global 宣告來告訴 TypeScript 我們的 ort 在 window 上
declare global {
    interface Window {
        ort: any;
    }
}

interface CameraViewProps {
    videoRef: RefObject<HTMLVideoElement | null>;
}

/**
 * 攝影機掃描視圖 (CameraView - ONNX 離線版)
 */
export function CameraView({ videoRef }: CameraViewProps) {
    const { addItem, tempDetections, clearTempDetections, settings } = useIngredients();
    const [isScanning, setIsScanning] = useState(false);
    const [scanMode, setScanMode] = useState<"local" | "cloud">("local");
    const [currentBoxes, setCurrentBoxes] = useState<any[]>([]);
    const [modelLoaded, setModelLoaded] = useState(false);
    const [sensitivity, setSensitivity] = useState(50); // 0-100, maps to 0.1-0.9 confidence
    const sessionRef = useRef<any>(null);

    // 類別名稱對照表 (由 yoloService 統一管理)
    const CLASS_NAMES = yoloService.CLASS_NAMES;

    // 初始化：檢查 yoloService 狀態
    useEffect(() => {
        if (yoloService.isLoaded()) {
            setModelLoaded(true);
            sessionRef.current = yoloService.getSession();
        } else {
            // 如果尚未預熱完成，則持續輪詢
            const timer = setInterval(() => {
                if (yoloService.isLoaded()) {
                    setModelLoaded(true);
                    sessionRef.current = yoloService.getSession();
                    clearInterval(timer);
                }
            }, 500);
            return () => clearInterval(timer);
        }

        // 偵測網路狀態並自動切換模式
        const handleOffline = () => {
            if (scanMode === "cloud") {
                setScanMode("local");
                notificationService.send("離線模式已啟動", "偵測到離線狀態，已自動切換至本地 YOLO 辨識模式。");
            }
        };
        window.addEventListener('offline', handleOffline);
        return () => window.removeEventListener('offline', handleOffline);
    }, []);

    const handleScan = async () => {
        hapticService.medium(); // 觸感震動回饋
        if (scanMode === "local") {
            await handleLocalScan();
        } else {
            await handleGeminiScan();
        }
    };

    const handleLocalScan = async () => {
        if (!videoRef.current || !sessionRef.current) return;
        setIsScanning(true);
        setCurrentBoxes([]);
        clearTempDetections();

        try {
            // ... (existing YOLO logic)
            const canvas = document.createElement("canvas");
            canvas.width = 640;
            canvas.height = 640;
            const ctx = canvas.getContext("2d");
            if (!ctx) return;
            ctx.drawImage(videoRef.current, 0, 0, 640, 640);

            const imgData = ctx.getImageData(0, 0, 640, 640);
            const input = new Float32Array(3 * 640 * 640);
            for (let i = 0; i < 640 * 640; i++) {
                input[i] = imgData.data[i * 4] / 255.0;
                input[i + 640 * 640] = imgData.data[i * 4 + 1] / 255.0;
                input[i + 2 * 640 * 640] = imgData.data[i * 4 + 2] / 255.0;
            }
            const tensor = new window.ort.Tensor("float32", input, [1, 3, 640, 640]);

            const feeds = { [sessionRef.current.inputNames[0]]: tensor };
            const results = await sessionRef.current.run(feeds);
            const outputView = results[sessionRef.current.outputNames[0]];
            const output = outputView.data as Float32Array;
            const dims = outputView.dims;

            const detections: any[] = [];
            // 滑桿值 0-100 對應到信心閾值 0.05~0.85（越低越靈敏）
            const CONF_THRESHOLD = 0.05 + (sensitivity / 100) * 0.80;

            const isTransposed = dims[1] > dims[2];
            const numAnchors = isTransposed ? dims[1] : dims[2];
            const numChannels = isTransposed ? dims[2] : dims[1];

            for (let i = 0; i < numAnchors; i++) {
                let maxConf = 0;
                let classId = -1;

                for (let c = 0; c < CLASS_NAMES.length; c++) {
                    const idx = isTransposed ? i * numChannels + (c + 4) : (c + 4) * numAnchors + i;
                    const conf = output[idx];
                    if (conf > maxConf) {
                        maxConf = conf;
                        classId = c;
                    }
                }

                if (maxConf > CONF_THRESHOLD) {
                    const cx = output[isTransposed ? i * numChannels : i];
                    const cy = output[isTransposed ? i * numChannels + 1 : numAnchors + i];
                    const w = output[isTransposed ? i * numChannels + 2 : numAnchors * 2 + i];
                    const h = output[isTransposed ? i * numChannels + 3 : numAnchors * 3 + i];

                    detections.push({
                        name: CLASS_NAMES[classId],
                        confidence: maxConf,
                        box: [(cx - w / 2) / 640, (cy - h / 2) / 640, (cx + w / 2) / 640, (cy + h / 2) / 640],
                        isSpoiled: CLASS_NAMES[classId].toLowerCase().includes("rotten"),
                        category: CLASS_NAMES[classId].includes("apple") || CLASS_NAMES[classId].includes("orange") || CLASS_NAMES[classId].includes("banana") ? "水果" : (CLASS_NAMES[classId].includes("cabbage") || CLASS_NAMES[classId].includes("spinach") ? "蔬菜" : (CLASS_NAMES[classId].includes("meat") ? "肉類" : "其他"))
                    });
                }
            }

            const nmsDetections: any[] = [];
            const sortedDetections = detections.sort((a, b) => b.confidence - a.confidence);
            for (const det of sortedDetections) {
                let keep = true;
                for (const kept of nmsDetections) {
                    const interX1 = Math.max(det.box[0], kept.box[0]);
                    const interY1 = Math.max(det.box[1], kept.box[1]);
                    const interX2 = Math.min(det.box[2], kept.box[2]);
                    const interY2 = Math.min(det.box[3], kept.box[3]);
                    const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
                    const areaA = (det.box[2] - det.box[0]) * (det.box[3] - det.box[1]);
                    const areaB = (kept.box[2] - kept.box[0]) * (kept.box[3] - kept.box[1]);
                    const iou = interArea / (areaA + areaB - interArea);
                    if (iou > 0.5) { keep = false; break; }
                }
                if (keep) {
                    nmsDetections.push(det);
                    if (nmsDetections.length >= 10) break;
                }
            }

            if (nmsDetections.length === 0) {
                notificationService.send("掃描完成", "未偵測到任何食材，請靠近一點或調整角度重試。");
            } else {
                setCurrentBoxes(nmsDetections);
                nmsDetections.forEach(det => addItem(det));
            }
        } catch (error) {
            console.warn("Local Scan Error:", error);
        } finally {
            setIsScanning(false);
        }
    };

    const handleGeminiScan = async () => {
        if (!videoRef.current) return;
        setIsScanning(true);
        setCurrentBoxes([]);
        clearTempDetections();

        try {
            // 擷取目前畫面
            const canvas = document.createElement("canvas");
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext("2d");
            if (!ctx) return;
            ctx.drawImage(videoRef.current, 0, 0);

            // 轉為 Base64 (JPEG)
            const base64Image = canvas.toDataURL("image/jpeg", 0.8).split(",")[1];

            // 呼叫 Gemini Vision
            const results = await llmService.detectIngredientsFromImage(base64Image);

            if (results.length === 0) {
                notificationService.send("掃描完成", "Gemini 未能在畫面中辨識出顯著食材。");
            } else {
                hapticService.light(); // 輕微觸感：檢測到物體
                results.forEach(item => addItem(item, "ai"));
                notificationService.send("掃描完成", `Gemini 成功辨識 ${results.length} 項食材`);
            }
        } catch (error: any) {
            console.error("Gemini Scan Error:", error);
            const msg = error.status ? `[Error ${error.status}] ${error.message}` : (error.message || "無法連線至 Gemini Vision");
            notificationService.send("掃描失敗", msg);
        } finally {
            setIsScanning(false);
        }
    };

    const handleClear = () => {
        clearTempDetections();
        setCurrentBoxes([]);
    };

    return (
        <div className="flex flex-col items-center w-full max-w-sm">
            <div className="relative w-full">
                {/* 掃描模式切換器 */}
                <div className="flex bg-[#0f2e24]/60 backdrop-blur-xl rounded-2xl p-1 mb-4 border border-primary/20 self-center w-fit mx-auto">
                    <button
                        onClick={() => setScanMode("local")}
                        className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${scanMode === "local" ? 'bg-primary text-[#0f2e24] shadow-[0_0_15px_rgba(0,255,136,0.3)]' : 'text-primary/60 hover:text-primary'}`}
                    >
                        <Brain size={14} />
                        YOLO 本地辨識
                    </button>
                    <button
                        onClick={() => setScanMode("cloud")}
                        className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${scanMode === "cloud" ? 'bg-primary text-[#0f2e24] shadow-[0_0_15px_rgba(0,255,136,0.3)]' : 'text-primary/60 hover:text-primary'}`}
                    >
                        <Sparkles size={14} />
                        Gemini 雲端辨識
                    </button>
                </div>

                {/* AI Status Badge */}
                <div className={`absolute top-16 left-1/2 transform -translate-x-1/2 z-20 bg-[#0f2e24]/80 backdrop-blur-md border ${!navigator.onLine && scanMode === "cloud" ? 'border-red-500' : !modelLoaded && scanMode === "local" ? 'border-red-400' : isScanning ? 'border-amber-400' : 'border-primary'} rounded-full px-4 py-1.5 flex items-center gap-2 shadow-[0_0_15px_rgba(0,255,136,0.3)] transition-colors duration-500`}>
                    <div className={`w-2 h-2 rounded-full ${!navigator.onLine && scanMode === "cloud" ? 'bg-red-500 animate-pulse' : !modelLoaded && scanMode === "local" ? 'bg-red-400' : isScanning ? 'bg-amber-400 animate-pulse' : 'bg-primary'} shadow-[0_0_8px_currentColor]`} />
                    <span className={`text-[10px] font-black tracking-widest ${!navigator.onLine && scanMode === "cloud" ? 'text-red-500' : !modelLoaded && scanMode === "local" ? 'text-red-400' : isScanning ? 'text-amber-400' : 'text-primary'} uppercase`}>
                        {!navigator.onLine && scanMode === "cloud" ? "偵測到離線：無法使用雲端 AI" : scanMode === "cloud" ? (isScanning ? "Cloud AI 串接中..." : "Gemini 視覺系統已就緒") : (!modelLoaded ? "系統核心啟動中..." : isScanning ? "正在為您辨識..." : "掃描系統已就緒")}
                    </span>
                </div>

                {/* 攝影機渲染區 (Camera Viewport) */}
                <div className="relative aspect-[4/5] sm:aspect-[3/4] rounded-[2.5rem] overflow-hidden bg-black border border-white/10 shadow-inner group">
                    {/* 精英級掃描線動效 */}
                    {isScanning && scanMode === "local" && (
                        <div className="animate-scan-laser" />
                    )}
                    
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className={`w-full h-full object-cover transition-all duration-700 ${isScanning ? 'scale-105 brightness-110' : 'scale-100 brightness-100'}`}
                    />

                    {/* Bounding Box Overlay */}
                    <div className="absolute inset-0 z-10 pointer-events-none">
                        {currentBoxes.map((boxData, idx) => boxData.box && (
                            <div
                                key={`box-${idx}`}
                                className="absolute"
                                style={{
                                    left: `${boxData.box[0] * 100}%`,
                                    top: `${boxData.box[1] * 100}%`,
                                    width: `${(boxData.box[2] - boxData.box[0]) * 100}%`,
                                    height: `${(boxData.box[3] - boxData.box[1]) * 100}%`,
                                    borderColor: boxData.isSpoiled ? '#ff4d4d' : 'var(--primary)',
                                    borderWidth: '2px',
                                    borderStyle: 'solid',
                                    borderRadius: '8px'
                                }}
                            >
                                <div className={`absolute -top-6 left-0 px-2 py-0.5 rounded-t-md text-[8px] font-black uppercase whitespace-nowrap ${boxData.isSpoiled ? 'bg-red-500 text-white' : 'bg-primary text-[#0f2e24]'}`}>
                                    {boxData.isSpoiled ? 'BAD' : 'GOOD'} | {boxData.name} | {Math.round((boxData.confidence || 0) * 100)}%
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="absolute inset-0 bg-gradient-to-b from-[#0f2e24]/40 to-transparent pointer-events-none" />

                    {isScanning && (
                        <div className="absolute inset-0 bg-primary/5 flex flex-col items-center justify-center">
                            <div className="w-full h-[2px] bg-amber-400 shadow-[0_0_15px_#fbbf24] absolute top-0 animate-[scan_2s_ease-in-out_infinite]" />
                        </div>
                    )}
                </div>
            </div>

            {tempDetections.length > 0 && (
                <div className="w-full flex justify-end px-2 mb-2 mt-4">
                    <button
                        onClick={handleClear}
                        className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 text-red-500 rounded-lg border border-red-500/20 hover:bg-red-500 hover:text-white transition-all text-[10px] font-black tracking-widest uppercase"
                    >
                        <RefreshCw size={12} />
                        重新整理畫面
                    </button>
                </div>
            )}


            <div className="w-full mt-8 space-y-3 px-2">
                <button
                    onClick={handleScan}
                    disabled={isScanning || (scanMode === "local" && !modelLoaded)}
                    className="w-full bg-primary text-background py-3 rounded-2xl font-black text-base flex items-center justify-center gap-2 hover:brightness-110 transition-all active:scale-[0.98] shadow-[0_8px_30px_var(--primary-glow)] disabled:opacity-50"
                >
                    {isScanning ? <Loader2 size={24} className="animate-spin" /> : scanMode === "cloud" ? <Sparkles size={24} strokeWidth={3} /> : <Camera size={24} strokeWidth={3} />}
                    <span className="truncate">{isScanning ? "正在分析介面..." : scanMode === "cloud" ? "Gemini 深度辨識" : "開始識別"}</span>
                </button>

                {/* 靈敏度滑桿 (僅 YOLO 本地模式顯示) */}
                {scanMode === "local" && (
                    <div className="bg-[#0f2e24]/60 backdrop-blur-xl rounded-2xl p-4 border border-primary/20">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                                <SlidersHorizontal size={13} className="text-primary" />
                                <span className="text-[10px] font-black tracking-widest uppercase text-primary/80">辨識靈敏度</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-[10px] font-black text-primary/50 uppercase">閾值</span>
                                <span className="text-[11px] font-black text-primary font-mono">{(0.05 + (sensitivity / 100) * 0.80).toFixed(2)}</span>
                            </div>
                        </div>
                        <div className="relative">
                            <input
                                id="sensitivity-slider"
                                type="range"
                                min={0}
                                max={100}
                                value={sensitivity}
                                onChange={(e) => setSensitivity(Number(e.target.value))}
                                className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                                style={{
                                    background: `linear-gradient(to right, #00ff88 ${sensitivity}%, rgba(0,255,136,0.15) ${sensitivity}%)`,
                                }}
                            />
                        </div>
                        <div className="flex justify-between mt-2">
                            <span className="text-[9px] text-primary/40 uppercase tracking-wider">高靈敏度 (較寬鬆)</span>
                            <span className="text-[9px] text-primary/40 uppercase tracking-wider">高精確度 (較嚴格)</span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
