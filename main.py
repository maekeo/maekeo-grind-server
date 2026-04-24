import cv2
import numpy as np
import base64
import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image: str  # base64 JPEG

@app.get("/")
def root():
    return {"status": "MAEKEO LAB 분쇄도 분석 서버 가동 중"}

@app.post("/analyze")
def analyze(req: ImageRequest):
    try:
        # 1. base64 → numpy 이미지
        img_bytes = base64.b64decode(req.image)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

        h, w = img.shape[:2]

        # 2. 동전 감지 (500원 동전 지름 26.5mm 기준)
        px_per_mm = detect_coin(img)

        # 3. 입자 분석
        particles = detect_particles(img)

        if len(particles) < 3:
            raise HTTPException(status_code=422, detail="입자를 충분히 감지하지 못했습니다. 흰 배경에 원두를 넓게 펼쳐 다시 촬영해주세요.")

        # 4. 크기 계산
        if px_per_mm:
            # 동전 기준 실측값
            sizes_um = [math.sqrt(a) / px_per_mm * 1000 for a in particles]
            method = "coin_calibrated"
        else:
            # 동전 없을 때 — 이미지 크기 대비 상대 추정
            # 평균 모카포트 분쇄 기준(400μm)으로 보정
            ref_area = np.median(particles)
            ref_um = 400
            sizes_um = [math.sqrt(a / ref_area) * ref_um for a in particles]
            method = "estimated"

        sizes_um = [s for s in sizes_um if 50 < s < 1500]  # 이상치 제거

        if not sizes_um:
            raise HTTPException(status_code=422, detail="유효한 입자를 감지하지 못했습니다.")

        avg_um    = float(np.mean(sizes_um))
        median_um = float(np.median(sizes_um))
        std_um    = float(np.std(sizes_um))
        min_um    = float(np.min(sizes_um))
        max_um    = float(np.max(sizes_um))

        # 5. 균일도 계산 (CV 기반, 낮을수록 균일)
        cv = (std_um / avg_um) * 100
        uniformity = max(0, min(100, round(100 - cv)))

        # 6. 분쇄 레벨 판정
        level_kor, level, percent = classify_grind(avg_um)

        # 7. 매커포트 적합도 (300~500μm 기준)
        moka_fit = classify_moka_fit(avg_um)

        # 8. 조언 생성
        advice = generate_advice(avg_um, moka_fit)

        return {
            "result": {
                "levelKor":     level_kor,
                "level":        level,
                "percent":      percent,
                "particleSize": round(avg_um),
                "medianSize":   round(median_um),
                "sizeMin":      round(min_um),
                "sizeMax":      round(max_um),
                "stdDev":       round(std_um),
                "uniformity":   uniformity,
                "particleCount": len(sizes_um),
                "mokaFit":      moka_fit,
                "bestBrew":     get_best_brew(avg_um),
                "advice":       advice,
                "calibrated":   px_per_mm is not None,
                "method":       method,
                "isCoffee":     True,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 오류: {str(e)}")


def detect_coin(img):
    """500원 동전(지름 26.5mm) 감지 → px/mm 반환. 없으면 None"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=w // 4,
        param1=50,
        param2=30,
        minRadius=int(w * 0.05),
        maxRadius=int(w * 0.25),
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # 가장 큰 원 = 동전으로 간주
        largest = max(circles, key=lambda c: c[2])
        radius_px = largest[2]
        diameter_px = radius_px * 2
        px_per_mm = diameter_px / 26.5
        return px_per_mm

    return None


def detect_particles(img):
    """원두 입자 윤곽선 감지 → 면적 리스트 반환"""
    # 흰 배경 기준으로 입자 분리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu 이진화 (자동 임계값)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 모폴로지 연산으로 잡음 제거
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 윤곽선 검출
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    img_area = h * w

    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 너무 작거나(노이즈) 너무 큰(배경) 것 제외
        if img_area * 0.00005 < area < img_area * 0.05:
            areas.append(area)

    return areas


def classify_grind(avg_um):
    """입자 크기 → 분쇄 레벨"""
    if avg_um < 200:
        return "극세분", "extra_fine", 5
    elif avg_um < 300:
        return "세분", "fine", 20
    elif avg_um < 500:
        return "중간", "medium", 45
    elif avg_um < 700:
        return "굵게", "coarse", 70
    else:
        return "아주 굵게", "extra_coarse", 90


def classify_moka_fit(avg_um):
    """매커포트 적합도 (최적: 300~500μm)"""
    if 300 <= avg_um <= 500:
        return "최적"
    elif 200 <= avg_um < 300:
        return "약간 고움"
    elif avg_um < 200:
        return "많이 고움"
    elif 500 < avg_um <= 650:
        return "약간 굵음"
    else:
        return "많이 굵음"


def get_best_brew(avg_um):
    if avg_um < 250:
        return "에스프레소"
    elif avg_um < 500:
        return "매커포트 / 모카포트"
    elif avg_um < 700:
        return "핸드드립 / 에어로프레스"
    else:
        return "프렌치프레스 / 콜드브루"


def generate_advice(avg_um, moka_fit):
    if moka_fit == "최적":
        return f"현재 분쇄도({round(avg_um)}μm)는 매커포트 최적 범위(300~500μm)입니다. 이 설정을 유지하세요."
    elif moka_fit == "약간 고움":
        return f"현재 분쇄도({round(avg_um)}μm)가 약간 고운 편입니다. 그라인더 설정을 1~2단계 굵게 조정하세요."
    elif moka_fit == "많이 고움":
        return f"현재 분쇄도({round(avg_um)}μm)가 너무 곱습니다. 과추출로 쓴맛이 강해질 수 있어요. 3~4단계 굵게 조정하세요."
    elif moka_fit == "약간 굵음":
        return f"현재 분쇄도({round(avg_um)}μm)가 약간 굵은 편입니다. 그라인더 설정을 1~2단계 곱게 조정하세요."
    else:
        return f"현재 분쇄도({round(avg_um)}μm)가 너무 굵습니다. 미추출로 싱거운 맛이 납니다. 3~4단계 곱게 조정하세요."
