import cv2
import numpy as np
import base64
import math
import gc
from PIL import Image
import io
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc.errors())})

class ImageRequest(BaseModel):
    image: str

@app.get("/")
def root():
    return {"status": "MAEKEO LAB 분쇄도 분석 서버 가동 중"}

@app.post("/analyze")
def analyze(req: ImageRequest):
    try:
        print(f"이미지 수신: {len(req.image)} chars")

        # 1. Decode & Resize
        img_bytes = base64.b64decode(req.image)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        del img_bytes

        MAX = 1000
        w, h = pil_img.size
        if max(w, h) > MAX:
            scale = MAX / max(w, h)
            pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        del pil_img
        h, w = img.shape[:2]
        print(f"이미지: {w}x{h}")

        # 2. 동전 감지 → px/mm
        px_per_mm = detect_coin(img, w, h)
        print(f"px/mm: {px_per_mm}")

        # 3. ROI + CLAHE + Adaptive Thresholding + Watershed
        markers, roi_h, roi_w = analyze_coffee_grounds(img)
        del img
        gc.collect()

        # 4. 입자 면적 측정
        roi_area = roi_h * roi_w
        particles, fines = extract_particles(markers, roi_area)
        del markers
        print(f"감지: 입자 {len(particles)}개, 미분 {len(fines)}개")

        if len(particles) < 1:
            return JSONResponse(content={
                "error": "입자를 감지하지 못했습니다. 흰 종이 위에 원두를 넓게 펼쳐 다시 촬영해주세요.",
                "particleCount": 0
            })

        # 5. 크기 계산
        sizes_um, fines_um, method = calc_sizes(particles, fines, px_per_mm)

        if not sizes_um:
            return JSONResponse(content={
                "error": "유효한 입자 크기를 계산하지 못했습니다. 다시 촬영해주세요.",
                "particleCount": len(particles)
            })

        avg_um     = float(np.mean(sizes_um))
        median_um  = float(np.median(sizes_um))
        std_um     = float(np.std(sizes_um))
        cv_val     = (std_um / avg_um) * 100 if avg_um > 0 else 0
        uniformity = max(0, min(100, round(100 - cv_val)))
        total      = len(sizes_um) + len(fines_um)
        fines_ratio = round(len(fines_um) / total * 100) if total > 0 else 0
        histogram  = build_histogram(sizes_um)
        level_kor, level, percent = classify_grind(avg_um)
        moka_fit   = classify_moka_fit(avg_um)
        advice     = generate_advice(avg_um, moka_fit, std_um, fines_ratio)

        print(f"완료: {level_kor} {round(avg_um)}μm ({len(sizes_um)}개)")

        return JSONResponse(content={"result": {
            "levelKor":      level_kor,
            "level":         level,
            "percent":       int(percent),
            "particleSize":  int(round(avg_um)),
            "medianSize":    int(round(median_um)),
            "stdDev":        int(round(std_um)),
            "uniformity":    int(uniformity),
            "particleCount": int(len(sizes_um)),
            "finesCount":    int(len(fines_um)),
            "finesRatio":    int(fines_ratio),
            "histogram":     histogram,
            "mokaFit":       moka_fit,
            "bestBrew":      get_best_brew(avg_um),
            "advice":        advice,
            "calibrated":    px_per_mm is not None,
            "method":        method,
            "isCoffee":      True,
        }})

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"분석 오류: {str(e)}")
    finally:
        gc.collect()


def detect_coin(img, w, h):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    del gray
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1,
        minDist=w//4, param1=60, param2=35,
        minRadius=int(w*0.05), maxRadius=int(w*0.30))
    del blur
    if circles is None:
        return None
    circles = np.round(circles[0]).astype("int")
    cx, cy, cr = max(circles, key=lambda c: c[2])
    diameter_px = cr * 2
    ratio = diameter_px / w
    print(f"동전: 지름={diameter_px}px 비율={ratio:.2f}")
    if not (0.08 <= ratio <= 0.45):
        print(f"동전 비율 이상({ratio:.2f})")
        return None
    return diameter_px / 24.0  # 100원 기준 24mm


def analyze_coffee_grounds(img):
    h, w = img.shape[:2]

    # ROI: 중앙 50% (렌즈 왜곡 회피)
    roi = img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    roi_h, roi_w = roi.shape[:2]
    print(f"ROI: {roi_w}x{roi_h}")

    # CLAHE (조명/그림자 평탄화)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    del gray

    # 가우시안 블러 (노이즈 제거)
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    del clahe_img

    # Adaptive Thresholding (국소 조명 변화 대응)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )
    del blurred

    # Morphology: 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    del thresh

    # Watershed: 겹친 입자 분리
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    del opening

    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    del dist_transform

    unknown = cv2.subtract(sure_bg, sure_fg)
    del sure_bg

    _, markers = cv2.connectedComponents(sure_fg)
    del sure_fg
    markers = markers + 1
    markers[unknown == 255] = 0
    del unknown

    markers = cv2.watershed(roi, markers)
    return markers, roi_h, roi_w


def extract_particles(markers, roi_area):
    particles, fines = [], []
    for label in np.unique(markers):
        if label <= 1:
            continue
        area = float(np.sum(markers == label))
        if area < roi_area * 0.000001: continue
        if area > roi_area * 0.05:     continue
        if area < roi_area * 0.00008:
            fines.append(area)
        else:
            particles.append(area)
    return particles, fines


def calc_sizes(particles, fines, px_per_mm):
    if px_per_mm:
        sizes_um = [math.sqrt(a) / px_per_mm * 1000 for a in particles]
        fines_um = [math.sqrt(a) / px_per_mm * 1000 for a in fines]
        method = "coin_calibrated"
    else:
        ref = float(np.median(particles)) if particles else 1
        sizes_um = [math.sqrt(a / ref) * 400 for a in particles]
        fines_um = [math.sqrt(a / ref) * 120 for a in fines]
        method = "estimated"

    sizes_um = [s for s in sizes_um if 30 < s < 3000]
    fines_um = [s for s in fines_um if 5  < s < 300]

    # 이상치 제거 (상하위 10%)
    if len(sizes_um) > 10:
        sizes_um = sorted(sizes_um)
        trim = max(1, int(len(sizes_um) * 0.10))
        sizes_um = sizes_um[trim:-trim]

    return sizes_um, fines_um, method


def build_histogram(sizes_um):
    bins   = [0,200,300,400,500,600,800,1000,99999]
    labels = ["<200","200-300","300-400","400-500","500-600","600-800","800-1000",">1000"]
    counts = [0] * 8
    for s in sizes_um:
        for i in range(len(bins)-1):
            if bins[i] <= s < bins[i+1]:
                counts[i] += 1; break
    return [{"range": labels[i], "count": counts[i]} for i in range(8)]

def classify_grind(avg_um):
    if avg_um < 200:   return "극세분",    "extra_fine",    5
    elif avg_um < 300: return "세분",      "fine",          20
    elif avg_um < 500: return "중간",      "medium",        45
    elif avg_um < 700: return "굵게",      "coarse",        70
    else:              return "아주 굵게", "extra_coarse",  90

def classify_moka_fit(avg_um):
    if 300 <= avg_um <= 500:  return "최적"
    elif 200 <= avg_um < 300: return "약간 고움"
    elif avg_um < 200:        return "많이 고움"
    elif 500 < avg_um <= 650: return "약간 굵음"
    else:                     return "많이 굵음"

def get_best_brew(avg_um):
    if avg_um < 250:   return "에스프레소"
    elif avg_um < 500: return "매커포트 / 모카포트"
    elif avg_um < 700: return "핸드드립 / 에어로프레스"
    else:              return "프렌치프레스 / 콜드브루"

def generate_advice(avg_um, moka_fit, std_um, fines_ratio):
    msgs = {
        "최적":     f"현재 분쇄도({round(avg_um)}μm)는 매커포트 최적 범위입니다.",
        "약간 고움": f"분쇄도({round(avg_um)}μm)가 약간 곱습니다. 1~2단계 굵게 조정하세요.",
        "많이 고움": f"분쇄도({round(avg_um)}μm)가 너무 곱습니다. 3~4단계 굵게 조정하세요.",
        "약간 굵음": f"분쇄도({round(avg_um)}μm)가 약간 굵습니다. 1~2단계 곱게 조정하세요.",
        "많이 굵음": f"분쇄도({round(avg_um)}μm)가 너무 굵습니다. 3~4단계 곱게 조정하세요.",
    }
    base = msgs.get(moka_fit, "")
    if std_um > 200:     base += " 입자 균일도가 낮습니다. 그라인더 버 점검을 권장합니다."
    if fines_ratio > 30: base += f" 미분 비율({fines_ratio}%)이 높습니다."
    return base
