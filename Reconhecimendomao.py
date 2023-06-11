import cv2
import mediapipe as mp

# Configurações
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
DETECTION_CONFIDENCE = 0.5

def redimensionar_imagem(imagem):
    return cv2.resize(imagem, (RESIZE_WIDTH, RESIZE_HEIGHT))

def detectar_desenhar_mao(imagem):
    resultado = reconhecimento_mao.process(imagem)

    if resultado.multi_hand_landmarks:
        for mao_landmarks in resultado.multi_hand_landmarks:
            desenho.draw_landmarks(imagem, mao_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                   desenho.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   desenho.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                  )

def main():
    webcam = cv2.VideoCapture(0)

    while webcam.isOpened():
        validacao, frame = webcam.read()
        if not validacao:
            break

        imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagem = redimensionar_imagem(imagem)

        detectar_desenhar_mao(imagem)

        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)
        cv2.imshow("Reconhecimento mao", imagem)

        if cv2.waitKey(5) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reconhecimento_mao = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=DETECTION_CONFIDENCE, min_tracking_confidence=DETECTION_CONFIDENCE)
    desenho = mp.solutions.drawing_utils
    main()
