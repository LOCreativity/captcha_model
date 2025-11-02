# captcha.py
import CaptchaCracker as cc
import os

WEIGHTS_PATH = 'weights.h5'  # 학습된 가중치 파일 경로
IMG_WIDTH = 130  # 타겟 이미지 너비
IMG_HEIGHT = 35  # 타겟 이미지 높이
IMG_LENGTH = 6  # 타겟 이미지 문자 수
IMG_CHAR = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}  # 타겟 이미지에 포함된 문자들
# ---------------------------------------------

try:
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"'{WEIGHTS_PATH}' 파일이 없습니다. train_captcha.py를 먼저 실행하세요.")

    AM = cc.ApplyModel(WEIGHTS_PATH, IMG_WIDTH, IMG_HEIGHT, IMG_LENGTH, IMG_CHAR)
    print("[Captcha] 모델 로드 성공.")
except Exception as e:
    print(f"[Captcha] 모델 로드 중 심각한 오류 발생: {e}")
    AM = None


def solve_captcha(target_img_path):
    if AM is None:
        print("[오류] Captcha 모델이 로드되지 않았습니다.")
        return None

    if not os.path.exists(target_img_path):
        print(f"[오류] 캡챠 이미지 파일을 찾을 수 없습니다: {target_img_path}")
        return None

    try:
        # 결과 도출
        pred = AM.predict(target_img_path)
        print(f"[Captcha] 예측 결과: {pred}")
        return pred
    except Exception as e:
        # TensorFlow가 파일을 못 찾는 경우
        if "Not found" in str(e):
            print(f"[오류] TensorFlow가 이미지 파일을 찾지 못했습니다: {target_img_path}")
        else:
            print(f"[오류] Captcha 예측 중 오류 발생: {e}")
        return None