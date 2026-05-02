import cv2
import numpy as np
import base64
import math
import json
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

        # 2. 이미지 품질 검사
        quality_issues, sharpness, brightness, coffee_ratio = check_image_quality(img)

        quality_warnings = []
        if "blur" in quality_issues:
            quality_warnings.append("초점이 흐립니다. 렌즈를 닦고 탭하여 초점을 맞춘 후 다시 촬영하세요.")
        if "dark" in quality_issues:
            quality_warnings.append("조명이 부족합니다. 더 밝은 환경에서 촬영하세요.")
        if "bright" in quality_issues:
            quality_warnings.append("사진이 너무 밝습니다. 직사광선을 피하고 다시 촬영하세요.")
        if "too_far" in quality_issues:
            quality_warnings.append("원두가 너무 작게 찍혔습니다. 더 가까이 촬영하세요.")

        # 3. 동전 감지
        px_per_mm = detect_coin(img)

        # 3. 입자 분석 (일반 입자 + 미분 분리)
        particles, fines = detect_particles_with_fines(img)

        if len(particles) < 3:
            raise HTTPException(status_code=422, detail="입자를 충분히 감지하지 못했습니다. 흰 배경에 원두를 넓게 펼쳐 다시 촬영해주세요.")

        # 4. 크기 계산
        if px_per_mm:
            sizes_um = [math.sqrt(a) / px_per_mm * 1000 for a in particles]
            fines_um = [math.sqrt(a) / px_per_mm * 1000 for a in fines]
            method = "coin_calibrated"
        else:
            ref_area = np.median(particles)
            ref_um = 400
            sizes_um = [math.sqrt(a / ref_area) * ref_um for a in particles]
            fines_um = [math.sqrt(a / ref_area) * ref_um * 0.3 for a in fines]
            method = "estimated"

        # 이상치 제거
        sizes_um = [s for s in sizes_um if 50 < s < 2000]
        fines_um = [s for s in fines_um if 10 < s < 150]

        if not sizes_um:
            raise HTTPException(status_code=422, detail="유효한 입자를 감지하지 못했습니다.")

        avg_um    = float(np.mean(sizes_um))
        median_um = float(np.median(sizes_um))
        std_um    = float(np.std(sizes_um))
        min_um    = float(np.min(sizes_um))
        max_um    = float(np.max(sizes_um))

        # 5. 균일도
        cv_val = (std_um / avg_um) * 100
        uniformity = max(0, min(100, round(100 - cv_val)))

        # 6. 미분 비율
        total_count = len(sizes_um) + len(fines_um)
        fines_ratio = round(len(fines_um) / total_count * 100) if total_count > 0 else 0

        # 7. 입자 분포 히스토그램 (구간별 개수)
        histogram = build_histogram(sizes_um)

        # 8. 분쇄 레벨 판정
        level_kor, level, percent = classify_grind(avg_um)
        moka_fit = classify_moka_fit(avg_um)
        advice = generate_advice(avg_um, moka_fit, std_um, fines_ratio)

        return {
            "result": {
                "levelKor":      level_kor,
                "level":         level,
                "percent":       percent,
                "particleSize":  round(avg_um),
                "medianSize":    round(median_um),
                "sizeMin":       round(min_um),
                "sizeMax":       round(max_um),
                "stdDev":        round(std_um),
                "uniformity":    uniformity,
                "particleCount": len(sizes_um),
                "finesCount":    len(fines_um),
                "finesRatio":    fines_ratio,
                "histogram":     histogram,
                "mokaFit":       moka_fit,
                "bestBrew":      get_best_brew(avg_um),
                "advice":        advice,
                "calibrated":    px_per_mm is not None,
                "method":        method,
                "qualityWarnings": quality_warnings,
                "sharpness":     round(sharpness, 1),
                "isCoffee":      True,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 오류: {str(e)}")


def detect_coin(img):
    """동전 감지 → px/mm 반환"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1,
        minDist=w // 4, param1=50, param2=30,
        minRadius=int(w * 0.05), maxRadius=int(w * 0.25),
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest = max(circles, key=lambda c: c[2])
        # 100원(24mm) 또는 500원(26.5mm) — 평균 25mm로 추정
        return (largest[2] * 2) / 25.0
    return None


def detect_particles_with_fines(img):
    """입자와 미분을 분리해서 감지"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    img_area = h * w

    particles = []  # 일반 입자
    fines = []      # 미분 (매우 작은 입자)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.000005:
            continue  # 노이즈 제거
        elif area < img_area * 0.0002:
            fines.append(area)      # 미분
        elif area < img_area * 0.05:
            particles.append(area)  # 일반 입자

    return particles, fines


def build_histogram(sizes_um):
    """입자 크기 분포 히스토그램 생성"""
    # 구간: 0~200, 200~300, 300~400, 400~500, 500~600, 600~800, 800~1000, 1000+
    bins = [0, 200, 300, 400, 500, 600, 800, 1000, 2000]
    labels = ["<200", "200-300", "300-400", "400-500", "500-600", "600-800", "800-1000", ">1000"]
    counts = [0] * (len(bins) - 1)

    for s in sizes_um:
        for i in range(len(bins) - 1):
            if bins[i] <= s < bins[i + 1]:
                counts[i] += 1
                break

    return [{"range": labels[i], "count": counts[i]} for i in range(len(labels))]


def classify_grind(avg_um):
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


def generate_advice(avg_um, moka_fit, std_um, fines_ratio):
    base = ""
    if moka_fit == "최적":
        base = f"현재 분쇄도({round(avg_um)}μm)는 매커포트 최적 범위(300~500μm)입니다."
    elif moka_fit == "약간 고움":
        base = f"분쇄도({round(avg_um)}μm)가 약간 고운 편입니다. 그라인더를 1~2단계 굵게 조정하세요."
    elif moka_fit == "많이 고움":
        base = f"분쇄도({round(avg_um)}μm)가 너무 곱습니다. 3~4단계 굵게 조정하세요."
    elif moka_fit == "약간 굵음":
        base = f"분쇄도({round(avg_um)}μm)가 약간 굵습니다. 그라인더를 1~2단계 곱게 조정하세요."
    else:
        base = f"분쇄도({round(avg_um)}μm)가 너무 굵습니다. 3~4단계 곱게 조정하세요."

    # 균일도 추가 조언
    if std_um > 200:
        base += " 입자 균일도가 낮습니다. 그라인더 버 점검을 권장합니다."

    # 미분 추가 조언
    if fines_ratio > 30:
        base += f" 미분 비율({fines_ratio}%)이 높습니다. 미분이 많으면 과추출로 쓴맛이 날 수 있습니다."

    return base


def check_image_quality(img):
    """이미지 품질 검사 — 흐림/어두움/원두 면적 체크"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    issues = []

    # 1. 흐림 감지 (Laplacian 분산)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        issues.append("blur")  # 초점 흐림

    # 2. 밝기 검사
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 60:
        issues.append("dark")  # 너무 어두움
    elif mean_brightness > 230:
        issues.append("bright")  # 너무 밝음 (과노출)

    # 3. 원두 면적 검사
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coffee_pixels = np.sum(binary > 0)
    coffee_ratio = coffee_pixels / (h * w)
    if coffee_ratio < 0.05:
        issues.append("too_far")  # 원두가 너무 작음

    return issues, laplacian_var, mean_brightness, coffee_ratio
