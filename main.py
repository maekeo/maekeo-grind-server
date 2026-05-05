import cv2
import numpy as np
import base64
import math
import gc
from PIL import Image
import io
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Maekerpot Grind Analyzer API")

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

# ════════════════════════════════════════════════════════════════
# BrewInsightEngine — 추출방식별 인사이트 생성기
# ════════════════════════════════════════════════════════════════
class BrewInsightEngine:
    def __init__(self):
        self.profiles = {
            "espresso":   {"min": 200, "max": 300, "max_fines": 15, "name": "에스프레소"},
            "maekerpot":  {"min": 350, "max": 500, "max_fines": 12, "name": "매커포트"},
            "pourover":   {"min": 600, "max": 800, "max_fines": 8,  "name": "핸드드립"},
            "frenchpress":{"min": 800, "max": 1200,"max_fines": 5,  "name": "프렌치프레스"},
        }

    def generate_insights(self, brew_method: str, actual_micron: float,
                          actual_fines: float, target_micron: int = None) -> dict:
        profile = self.profiles.get(brew_method, self.profiles["maekerpot"])
        target  = target_micron or int((profile["min"] + profile["max"]) / 2)

        status  = "perfect"
        title   = f"{profile['name']}에 완벽한 분쇄도입니다!"
        actions = []

        # 1. 평균 입도 분석
        diff = actual_micron - target
        if actual_micron < profile["min"]:
            status = "warning"
            title  = "입자가 너무 곱게 분쇄되었습니다."
            adj    = round((profile["min"] - actual_micron) / 50)
            actions.append(f"과추출로 인한 쓴맛이 날 수 있습니다. 그라인더를 {max(1,adj)}~{max(2,adj+1)}단계 굵게 조정하세요. (목표: {target}μm)")
        elif actual_micron > profile["max"]:
            status = "warning"
            title  = "입자가 너무 굵게 분쇄되었습니다."
            adj    = round((actual_micron - profile["max"]) / 50)
            actions.append(f"압력이 제대로 걸리지 않아 묽은 커피가 나올 수 있습니다. 그라인더를 {max(1,adj)}~{max(2,adj+1)}단계 곱게 조정하세요. (목표: {target}μm)")

        # 2. 미분 분석
        if actual_fines > profile["max_fines"]:
            prev_status = status
            status = "alert" if status == "warning" else "warning"
            if prev_status == "perfect":
                title = "주의: 미분 발생량이 기준치를 초과했습니다."
            actions.append(f"미분이 {actual_fines}%로 기준치({profile['max_fines']}%)를 초과했습니다. 추출 후반부에 떫은맛이 날 수 있으니 추출을 5초 일찍 끊어보세요.")
            actions.append("그라인더 버(Burr) 청소 주기가 도래했을 수 있습니다. 청소 후 재분쇄를 권장합니다.")

        # 3. 정상일 때
        if not actions:
            actions.append(f"현재 분쇄도({round(actual_micron)}μm)가 {profile['name']} 최적 범위({profile['min']}~{profile['max']}μm) 내에 있습니다.")
            actions.append("바로 추출을 진행하셔도 좋습니다.")

        return {
            "status":       status,   # perfect / warning / alert
            "title":        title,
            "action_items": actions,
        }

insight_engine = BrewInsightEngine()

# ════════════════════════════════════════════════════════════════
# API
# ════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "MAEKEO LAB 분쇄도 분석 서버 가동 중"}

@app.post("/analyze")
async def analyze(
    file:         UploadFile = File(...),
    brew_method:  str = "maekerpot",
    grinder_type: str = "flat_burr",
    target_micron:int = 400,
):
    try:
        contents = await file.read()
        print(f"이미지 수신: {len(contents)} bytes | brew={brew_method} target={target_micron}μm")

        nparr = np.frombuffer(contents, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del contents, nparr

        if img is None:
            return JSONResponse(content={"status":"error","message":"이미지 디코딩 실패"}, status_code=400)

        # 리사이즈
        MAX = 1000
        h, w = img.shape[:2]
        if max(h, w) > MAX:
            scale = MAX / max(h, w)
            img   = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            h, w  = img.shape[:2]
        print(f"이미지: {w}x{h}")

        # 파이프라인
        px_per_mm                    = detect_coin(img, w, h)
        markers, roi, roi_y, roi_x   = analyze_coffee_grounds(img)
        roi_h, roi_w                 = roi.shape[:2]
        particles, fines             = extract_particles(markers, roi_h * roi_w)
        overlay_b64                  = make_overlay(img, markers, roi, roi_y, roi_x)
        del markers, roi
        gc.collect()

        print(f"감지: 입자 {len(particles)}개, 미분 {len(fines)}개")

        if len(particles) < 1:
            return JSONResponse(content={
                "status": "error",
                "message": "입자를 감지하지 못했습니다. 흰 종이 위에 원두를 넓게 펼쳐 다시 촬영해주세요."
            })

        sizes_um, fines_um, method = calc_sizes(particles, fines, px_per_mm)
        if not sizes_um:
            return JSONResponse(content={
                "status": "error",
                "message": "유효한 입자 크기를 계산하지 못했습니다. 다시 촬영해주세요."
            })

        avg_um      = float(np.mean(sizes_um))
        median_um   = float(np.median(sizes_um))
        std_um      = float(np.std(sizes_um))
        cv_val      = (std_um / avg_um) * 100 if avg_um > 0 else 0
        uniformity  = max(0, min(100, round(100 - cv_val)))
        total       = len(sizes_um) + len(fines_um)
        fines_ratio = round(len(fines_um) / total * 100) if total > 0 else 0

        level_kor, level, percent = classify_grind(avg_um)
        moka_fit                  = classify_moka_fit(avg_um)
        histogram                 = build_histogram(sizes_um)

        # BrewInsightEngine
        insights = insight_engine.generate_insights(
            brew_method, avg_um, fines_ratio, target_micron
        )

        print(f"완료: {level_kor} {round(avg_um)}μm | {insights['status']}")

        return JSONResponse(content={
            "status": "success",
            "data": {
                "summary": {
                    "average_micron":   int(round(avg_um)),
                    "median_micron":    int(round(median_um)),
                    "std_dev":          int(round(std_um)),
                    "fines_percentage": int(fines_ratio),
                    "uniformity_score": int(uniformity),
                    "particle_count":   int(len(sizes_um)),
                    "fines_count":      int(len(fines_um)),
                },
                "classification": {
                    "levelKor":  level_kor,
                    "level":     level,
                    "percent":   int(percent),
                    "mokaFit":   moka_fit,
                    "bestBrew":  get_best_brew(avg_um),
                },
                "insights": insights,
                "distribution_chart": histogram,
                "visual_result": {
                    "overlay_image_b64": overlay_b64,
                },
                "meta": {
                    "calibrated":   px_per_mm is not None,
                    "method":       method,
                    "brew_method":  brew_method,
                    "grinder_type": grinder_type,
                    "target_micron":target_micron,
                    "isCoffee":     True,
                }
            }
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(content={"status":"error","message":f"분석 오류: {str(e)}"}, status_code=500)
    finally:
        gc.collect()


# ════════════════════════════════════════════════════════════════
# OpenCV 파이프라인
# ════════════════════════════════════════════════════════════════
def detect_coin(img, w, h):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    del gray
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1,
        minDist=w//4, param1=60, param2=35,
        minRadius=int(w*0.05), maxRadius=int(w*0.30))
    del blur
    if circles is None: return None
    circles = np.round(circles[0]).astype("int")
    cx, cy, cr = max(circles, key=lambda c: c[2])
    diameter_px = cr * 2
    ratio = diameter_px / w
    print(f"동전: 지름={diameter_px}px 비율={ratio:.2f}")
    if not (0.08 <= ratio <= 0.45): return None
    return diameter_px / 24.0

def analyze_coffee_grounds(img):
    h, w = img.shape[:2]
    roi_y, roi_x = int(h*0.25), int(w*0.25)
    roi = img[roi_y:int(h*0.75), roi_x:int(w*0.75)].copy()
    print(f"ROI: {roi.shape[1]}x{roi.shape[0]}")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray); del gray
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0); del clahe_img
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    del blurred
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2); del thresh
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5); del opening
    _, sure_fg = cv2.threshold(dist, 0.2*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg); del dist
    unknown = cv2.subtract(sure_bg, sure_fg); del sure_bg
    _, markers = cv2.connectedComponents(sure_fg); del sure_fg
    markers = markers + 1
    markers[unknown == 255] = 0; del unknown
    markers = cv2.watershed(roi, markers)
    return markers, roi, roi_y, roi_x

def extract_particles(markers, roi_area):
    particles, fines = [], []
    for label in np.unique(markers):
        if label <= 1: continue
        area = float(np.sum(markers == label))
        if area < roi_area * 0.000001: continue
        if area > roi_area * 0.05:     continue
        if area < roi_area * 0.00008:  fines.append(area)
        else:                          particles.append(area)
    return particles, fines

def make_overlay(img, markers, roi, roi_y, roi_x):
    overlay     = img.copy()
    roi_h, roi_w = roi.shape[:2]
    overlay_roi = overlay[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_area    = roi_h * roi_w
    for label in np.unique(markers):
        if label <= 1: continue
        area = float(np.sum(markers == label))
        if area < roi_area*0.000001 or area > roi_area*0.05: continue
        mask = np.uint8(markers == label)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        color = (0, 140, 255) if area < roi_area*0.00008 else (0, 220, 80)
        cv2.drawContours(overlay_roi, cnts, -1, color, 1)
    cv2.rectangle(overlay, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255,255,0), 2)
    ratio = 400 / max(overlay.shape[:2])
    small = cv2.resize(overlay, (int(overlay.shape[1]*ratio), int(overlay.shape[0]*ratio)))
    del overlay
    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70]); del small
    return base64.b64encode(buf).decode('utf-8')

def calc_sizes(particles, fines, px_per_mm):
    if px_per_mm:
        sizes_um = [math.sqrt(a)/px_per_mm*1000 for a in particles]
        fines_um = [math.sqrt(a)/px_per_mm*1000 for a in fines]
        method   = "coin_calibrated"
    else:
        ref      = float(np.median(particles)) if particles else 1
        sizes_um = [math.sqrt(a/ref)*400 for a in particles]
        fines_um = [math.sqrt(a/ref)*120 for a in fines]
        method   = "estimated"
    sizes_um = [s for s in sizes_um if 30 < s < 3000]
    fines_um = [s for s in fines_um if 5  < s < 300]
    if len(sizes_um) > 10:
        sizes_um = sorted(sizes_um)
        trim = max(1, int(len(sizes_um)*0.10))
        sizes_um = sizes_um[trim:-trim]
    return sizes_um, fines_um, method

def build_histogram(sizes_um):
    bins   = [0,100,200,300,400,500,600,99999]
    labels = ["<100","100-200","200-300","300-400","400-500","500-600","600+"]
    counts = [0]*7
    total  = len(sizes_um)
    for s in sizes_um:
        for i in range(len(bins)-1):
            if bins[i] <= s < bins[i+1]: counts[i]+=1; break
    values = [round(c/total*100,1) if total>0 else 0 for c in counts]
    return {"labels":labels,"values":values,"counts":counts}

def classify_grind(avg_um):
    if avg_um < 200:   return "극세분",    "extra_fine",   5
    elif avg_um < 300: return "세분",      "fine",         20
    elif avg_um < 500: return "중간",      "medium",       45
    elif avg_um < 700: return "굵게",      "coarse",       70
    else:              return "아주 굵게", "extra_coarse", 90

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
