import glob
import CaptchaCracker as cc
import os

# --- 설정값 ---
SAMPLE_DIR = 'sample'  # 샘플 이미지가 있는 폴더
SAMPLE_PATTERN = 'sample/*.png'  # 학습 데이터 이미지 경로 (파일명이 정답이어야 함)
IMG_WIDTH = 130  # 이미지 너비
IMG_HEIGHT = 35  # 이미지 높이
EPOCHS = 300  # 반복 학습 횟수 (100만 해도 될듯)
OUTPUT_WEIGHTS = 'weights.h5'  # 저장될 가중치 파일명


# -------------

def learn_img():
    print("캡챠 모델 학습을 시작합니다...")

    # 샘플 폴더 확인
    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)
        print(f"[경고] '{SAMPLE_DIR}' 폴더가 없어 새로 생성.")
        return

    img_path_list = glob.glob(SAMPLE_PATTERN)

    if not img_path_list:
        print(f"[오류] '{SAMPLE_PATTERN}'에 해당하는 학습 이미지가 없습니다.")
        print(f"'{SAMPLE_DIR}' 폴더에 샘플 이미지를 추가하세요.")
        return

    print(f"총 {len(img_path_list)}개의 샘플 이미지로 학습을 시작합니다.")

    try:
        # 학습모델 생성
        CM = cc.CreateModel(img_path_list, IMG_WIDTH, IMG_HEIGHT)

        # 반복 학습 시작
        model = CM.train_model(epochs=EPOCHS)

        # 학습 결과 가중치 저장
        model.save_weights(OUTPUT_WEIGHTS)

        print(f"\n학습 완료! '{OUTPUT_WEIGHTS}' 파일이 저장되었습니다.")

    except Exception as e:
        print(f"모델 학습 중 오류 발생: {e}")
        print("TensorFlow, CUDA, cuDNN 등이 올바르게 설치되었는지 확인하세요.")


if __name__ == "__main__":
    learn_img()