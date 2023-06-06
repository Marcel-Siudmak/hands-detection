import cv2
import mediapipe as mp

# Inicjalizacja modułu MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    # Inicjalizacja kamery internetowej
    cap = cv2.VideoCapture(0)
    
    # Inicjalizacja obiektu detekcji dłoni
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.9) as hands:
        while True:
            # Odczytaj ramkę z kamery
            ret, frame = cap.read()

            # Odbicie lustrzane
            frame = cv2.flip(frame, 1)
            
            # Konwersja ramki na obiekt RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Wykonaj detekcję dłoni
            results = hands.process(frame_rgb)
            
            # Sprawdź, czy wykryto dłoń
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Pobierz pozycje czubka palca wskazującego
                    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    finger_x = int(index_finger.x * frame.shape[1])
                    finger_y = int(index_finger.y * frame.shape[0])
                    
                    # Wyświetl pozycję czubka palca
                    cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 0), -1)
                    print("x:", finger_x, "y:", finger_y)
                    
            # Wyświetl ramkę z oznaczoną pozycją palca
            cv2.imshow('Finger Tracking', frame)
            
            # Wyjście z pętli, jeśli naciśnięto klawisz 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Zwolnienie zasobów
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
