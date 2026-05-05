import cv2
import numpy as np
import base64
import math
import gc
import json
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
    body = await request.body()
    print(f"422 검증 오류: {exc.errors()}")
    print(f"요청 바디 첫 100자: {body[:100]}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc.errors())}
    )

class ImageRequest(BaseModel):
    image: str

@app.get("/")
def root():
    return {"status": "MAEKEO LAB 분쇄도 분석 서버 가동 중"}

@app.post("/analyze")
def analyze(req: ImageRequest):
    try:
        print(f"이미지 수신: {len(req.image)} chars")

        # 1. base64 → PIL 이미지로 먼저 디코딩 (메모리 효율적)
        img_bytes = base64.b64decode(req.image)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # 2. 600px로 리사이즈 (메모리 절약)
        MAX = 800
        w, h = pil_img.size
        if max(w, h) > MAX:
            scale = MAX / max(w, h)
            pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        # 3. OpenCV 배열로 변환
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        del pil_img, img_bytes
        gc.collect()

        h, w = img.shape[:2]
        print(f"이미지 크기: {w}x{h}")

        # 4. 동전 감지
        px_per_mm = detect_coin(img)
        print(f"동전 감지: {px_per_mm}")

        # 5. 입자 분석
        particles, fines, h, w = detect_particles(img)
        print(f"입자: {len(particles)}개, 미분: {len(fines)}개")

        if len(particles) < 1:
            print(f"입자 부족: {len(particles)}개")
            return JSONResponse(content={
                "error": "입자를 감지하지 못했습니다. 흰 배경에 원두를 넓게 펼쳐 다시 촬영해주세요.",
                "particleCount": len(particles)
            }, status_code=200)

        # 6. 크기 계산
        if px_per_mm:
            sizes_um = [math.sqrt(a) / px_per_mm * 1000 for a in particles]
            fines_um = [math.sqrt(a) / px_per_mm * 1000 for a in fines]
            method = "coin_calibrated"
        else:
            # 추정: 이미지 크기 기반 상대 크기 계산
            # 600px 이미지에서 중간 분쇄도(400μm) 입자는 약 10~15px 지름
            img_scale = max(h, w) / 600.0
            px_per_mm_est = 25.0 / img_scale  # 경험적 기준값
            sizes_um = [math.sqrt(a) / px_per_mm_est * 1000 for a in particles]
            fines_um = [math.sqrt(a) / px_per_mm_est * 1000 for a in fines]
            method = "estimated"

        sizes_um = [s for s in sizes_um if 30 < s < 3000]
        fines_um = [s for s in fines_um if 5 < s < 200]

        # 필터 후 비어있으면 원본 사용
        if not sizes_um and particles:
            if px_per_mm:
                sizes_um = [math.sqrt(a) / px_per_mm * 1000 for a in particles]
            else:
                ref_area = np.median(particles)
                sizes_um = [math.sqrt(a / ref_area) * 400 for a in particles]
            print(f"필터 완화 후 입자: {len(sizes_um)}개")

        if not sizes_um:
            print("유효 입자 없음")
            return JSONResponse(content={
                "error": "유효한 입자를 감지하지 못했습니다.",
                "particleCount": 0
            }, status_code=200)

        avg_um    = float(np.mean(sizes_um))
        std_um    = float(np.std(sizes_um))
        cv_val    = (std_um / avg_um) * 100
        uniformity = max(0, min(100, round(100 - cv_val)))
        total_count = len(sizes_um) + len(fines_um)
        fines_ratio = round(len(fines_um) / total_count * 100) if total_count > 0 else 0
        histogram = build_histogram(sizes_um)
        level_kor, level, percent = classify_grind(avg_um)
        moka_fit = classify_moka_fit(avg_um)
        advice = generate_advice(avg_um, moka_fit, std_um, fines_ratio)

        del img
        gc.collect()

        result_data = {
            "levelKor": str(level_kor),
            "level": str(level),
            "percent": int(percent),
            "particleSize": int(round(avg_um)),
            "stdDev": int(round(std_um)),
            "uniformity": int(uniformity),
            "particleCount": int(len(sizes_um)),
            "finesCount": int(len(fines_um)),
            "finesRatio": int(fines_ratio),
            "histogram": histogram,
            "mokaFit": str(moka_fit),
            "bestBrew": str(get_best_brew(avg_um)),
            "advice": str(advice),
            "calibrated": bool(px_per_mm is not None),
            "method": str(method),
            "isCoffee": True,
        }
        print(f"분석 완료: {result_data['levelKor']} {result_data['particleSize']}μm")
        return JSONResponse(content={"result": result_data})

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"분석 오류: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"분석 오류: {str(e)}")
    finally:
        gc.collect()


def detect_coin(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = blur.shape
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1,
        minDist=w//4, param1=50, param2=30,
        minRadius=int(w*0.05), maxRadius=int(w*0.22))
    del gray, blur
    if circles is not None:
        circles = np.round(circles[0]).astype("int")
        largest = max(circles, key=lambda c: c[2])
        radius = largest[2]
        diameter_px = radius * 2
        ratio = diameter_px / w
        print(f"동전 비율: {ratio:.2f} (지름 {diameter_px}px / 이미지 {w}px)")
        # 동전이 이미지 너비의 10~40% 범위일 때만 유효
        if 0.10 <= ratio <= 0.40:
            return diameter_px / 24.0  # 100원 기준 24mm
        else:
            print(f"동전 비율 이상({ratio:.2f}) → 추정 방식 사용")
            return None
    return None


def detect_particles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    del img
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    del gray

    # thresh=80 고정 (배경이 밝고 원두가 어두운 경우 최적)
    _, binary = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
    del blur

    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    del binary

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = cleaned.shape
    del cleaned
    img_area = h * w

    particles, fines = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.000001: continue  # 노이즈 제거
        if area > img_area * 0.01: continue       # 너무 큰 덩어리 제외
        if area < img_area * 0.0001:
            fines.append(area)
        else:
            particles.append(area)

    print(f"감지: 입자 {len(particles)}개, 미분 {len(fines)}개")
    return particles, fines, h, w


def build_histogram(sizes_um):
    bins = [0, 200, 300, 400, 500, 600, 800, 1000, 99999]
    labels = ["<200","200-300","300-400","400-500","500-600","600-800","800-1000",">1000"]
    counts = [0] * 8
    for s in sizes_um:
        for i in range(len(bins)-1):
            if bins[i] <= s < bins[i+1]:
                counts[i] += 1; break
    return [{"range": labels[i], "count": counts[i]} for i in range(8)]


def classify_grind(avg_um):
    if avg_um < 200: return "극세분", "extra_fine", 5
    elif avg_um < 300: return "세분", "fine", 20
    elif avg_um < 500: return "중간", "medium", 45
    elif avg_um < 700: return "굵게", "coarse", 70
    else: return "아주 굵게", "extra_coarse", 90


def classify_moka_fit(avg_um):
    if 300 <= avg_um <= 500: return "최적"
    elif 200 <= avg_um < 300: return "약간 고움"
    elif avg_um < 200: return "많이 고움"
    elif 500 < avg_um <= 650: return "약간 굵음"
    else: return "많이 굵음"


def get_best_brew(avg_um):
    if avg_um < 250: return "에스프레소"
    elif avg_um < 500: return "매커포트 / 모카포트"
    elif avg_um < 700: return "핸드드립 / 에어로프레스"
    else: return "프렌치프레스 / 콜드브루"


def generate_advice(avg_um, moka_fit, std_um, fines_ratio):
    if moka_fit == "최적": base = f"현재 분쇄도({round(avg_um)}μm)는 매커포트 최적 범위입니다."
    elif moka_fit == "약간 고움": base = f"분쇄도({round(avg_um)}μm)가 약간 곱습니다. 1~2단계 굵게 조정하세요."
    elif moka_fit == "많이 고움": base = f"분쇄도({round(avg_um)}μm)가 너무 곱습니다. 3~4단계 굵게 조정하세요."
    elif moka_fit == "약간 굵음": base = f"분쇄도({round(avg_um)}μm)가 약간 굵습니다. 1~2단계 곱게 조정하세요."
    else: base = f"분쇄도({round(avg_um)}μm)가 너무 굵습니다. 3~4단계 곱게 조정하세요."
    if std_um > 200: base += " 입자 균일도가 낮습니다. 그라인더 버 점검을 권장합니다."
    if fines_ratio > 30: base += f" 미분 비율({fines_ratio}%)이 높습니다."
    return base
