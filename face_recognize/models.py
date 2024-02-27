from django.db import models
import cv2
import dlib
import numpy as np
# Create your models here.
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean


class Badpeople(models.Model):
    Male = 'male'
    Female = 'female'
    SexChoise = [
        (Male, 'male'),
        (Female, 'female'),
    ]
    name = models.CharField(max_length=30)
    age = models.IntegerField()
    feature = models.JSONField(unique=True, blank=True, null=True)
    sex = models.CharField(max_length=30, choices=SexChoise, default=Male)
    image = models.ImageField(upload_to='badpeople/')
    '''
    def create_feature(self):
        detector = dlib.get_frontal_face_detector()
        # Загрузка изображения
        image = cv2.imread(self.image.path)

        # Преобразование изображения в монохромное для улучшения производительности
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц на изображении с помощью детектора лиц из библиотеки dlib
        faces = detector(gray)
        for face in faces:
            # Получение координат прямоугольника, описывающего лицо
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # Вырезание изображения лица из исходного изображения
            face_image = image[y:y+h, x:x+w]
            # Извлечение характеристик лица
            self.feature = extract_face_features(face_image)
        self.save()
    '''     
    def create_feature(self):
        # Загрузка изображения с помощью OpenCV
        image = cv2.imread(self.image.path)
        if image is None:
            print("Ошибка: Невозможно загрузить изображение.")
            return None
    
        # Извлечение характеристик лица
        self.feature = extract_face_features(image)
        self.save()

    def save(self, *args, **kwargs):
        # Если объект еще не сохранен в базу данных
        if not self.pk:
            # Сначала сохраняем объект, чтобы получить его первичный ключ
            super().save(*args, **kwargs)
            # Затем записываем характеристики
            self.create_feature()
        else:
            super().save(*args, **kwargs) 
        


def compare_features(features1, features2):
    # Проверяем, являются ли аргументы массивами numpy
    features1 = np.array(features1)
    features2 = np.array(features2)
    # Вычисляем евклидово расстояние между признаками
    distance = euclidean(features1.flatten(), features2.flatten())
    # Инвертируем расстояние, так как меньшее расстояние соответствует большему сходству
    similarity = 1 / (1 + distance)
    return similarity

'''
def compare_features(features1, features2):
    # Проверяем, являются ли аргументы массивами numpy
    if isinstance(features1, np.ndarray) and isinstance(features2, np.ndarray):
        # Проверяем наличие NaN значений в признаках
        if not np.isnan(features1).any() and not np.isnan(features2).any():
            # Вычисляем косинусное сходство между признаками
            similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
            return similarity
        else:
            # Если есть NaN значения, возвращаем нулевое сходство
            return 0.0
    else:
        # Если аргументы не являются массивами numpy, возвращаем None
        return None



def compare_features(features1, features2):
    features1 = np.array(features1)
    features2 = np.array(features2)
    # Вычислите евклидово расстояние между двумя векторами характеристик
    distance = np.linalg.norm(features1 - features2)
    # Чем меньше расстояние, тем больше сходство
    similarity = 1 / (1 + distance)
    return similarity

# Функция для извлечения характеристик (черт лица) из изображения лица
def extract_face_features(face_image):
    if face_image is None:
        print("Ошибка: Изображение лица не найдено.")
        return None
    shape_predictor = dlib.shape_predictor('/home/myrza/computer_vision/face_recognize/shape_predictor_68_face_landmarks.dat')
    
    # Преобразование изображения в монохромное для детектора
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    # Обнаружение ключевых точек лица (черты лица)
    shape = shape_predictor(gray, dlib.rectangle(0, 0, face_image.shape[1], face_image.shape[0]))
    # Преобразование формата ключевых точек в список координат
    face_features = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return face_features'''

def extract_face_features(face_image):
    # Инициализация модели распознавания лиц
    face_rec_model = dlib.face_recognition_model_v1("/home/myrza/computer_vision/face_recognize/dlib_face_recognition_resnet_model_v1.dat")
    
    # Преобразование изображения в формат RGB
    image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Обнаружение лиц на изображении
    detector = dlib.get_frontal_face_detector()
    face_detections = detector(image_rgb)
    
    # Проверка наличия лиц на изображении
    if not face_detections:
        print("На изображении не найдено лиц.")
        return None
    
    face_features_list = []
    
    # Извлечение характеристик для каждого обнаруженного лица
    for face in face_detections:
        # Создание объекта full_object_detection для обнаруженного лица
        shape_predictor = dlib.shape_predictor("/home/myrza/computer_vision/face_recognize/shape_predictor_68_face_landmarks.dat")
        full_object_detection = shape_predictor(image_rgb, face)  # Используем shape_predictor для получения full_object_detection
        
        # Извлечение характеристик лица с использованием модели ResNet
        face_descriptor = face_rec_model.compute_face_descriptor(image_rgb, full_object_detection)
        
        # Преобразование формата характеристик в список
        face_features = list(face_descriptor)
        
        face_features_list.append(face_features)
    
    return face_features_list


def detect_faces(image_path):
    # Загрузка каскадного классификатора для детекции лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Преобразование изображения в монохромное
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Детекция лиц на изображении
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Возвращаем обнаруженные лица
    return faces


def get_face(image_path):
    male_badpeople = [badpeople.feature for badpeople in Badpeople.objects.filter(sex='male')]
    female_badpeople = [badpeople.feature for badpeople in Badpeople.objects.filter(sex='female')]
    all_badpeople_features = [person.feature for person in Badpeople.objects.all()]
    # Инициализация детектора лиц и загрузка модели распознавания лиц
    detector = dlib.get_frontal_face_detector()
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование изображения в монохромное для улучшения производительности
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Обнаружение лиц на изображении с помощью детектора лиц из библиотеки dlib
    faces = detect_faces(image_path)
    print(faces)
    print(len(faces))
    print('Лицо')
    countoffaces = len(faces)
    if len(faces) == 0:
        print("На изображении нет обнаруженных лиц.")
        return None

    for face in faces:
        # Извлечение изображения лица
        print(face)
        face_image = image[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]

        # Получение пола лица
        gender = get_gender(image)

        # Извлечение характеристик лица
        face_features = extract_face_features(face_image)

        # Проверка наличия лица в базе плохих людей
        if gender == 'Female':
            for badperson in female_badpeople:
                print(badperson)
                if face_features is None:
                    continue
                similarity = compare_features(badperson, face_features)
                print('Femalesimilarity')
                print(similarity)
                if similarity is not None and similarity > 0.92-countoffaces*0.1:
                    print('рядом плохой человек')
        elif gender == 'Male':
            for badperson in male_badpeople:
                if face_features is None:
                    continue
                print(badperson)
                similarity = compare_features(badperson, face_features)
                print('Malesimilarity')
                print(similarity)
                if similarity is not None and similarity > 0.92-countoffaces*0.1:
                    print('рядом плохой человек')
        else:
            for badperson in all_badpeople_features:
                if face_features is None:
                    continue
                similarity = compare_features(badperson, face_features)
                print('все similarity')
                if similarity is not None and similarity > 0.92-countoffaces*0.1:
                    print('рядом плохой человек')


def get_gender(face_image):
    # Загрузка предварительно обученной модели для классификации пола
    gender_model = cv2.dnn.readNetFromCaffe('/home/myrza/computer_vision/face_recognize/deploy_gender.prototxt', '/home/myrza/computer_vision/face_recognize/gender_net.caffemodel')

    # Преобразование области лица в формат, необходимый для классификации гендера
    blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Применение модели для классификации гендера
    gender_model.setInput(blob)
    gender_preds = gender_model.forward()

    # Получение метки класса с наибольшей вероятностью
    gender = 'Male' if gender_preds[0][0] > 0.5 else 'Female'

    return gender


print(get_face('/home/myrza/computer_vision/media/check/три.jpeg'))



'''       \\\ ЭТА ФУНКЦИЯ НАМНОГО СЛАБЕЕ НО ЭФФЕКТИВНЕЕ  \\\\
def get_face(image):
    # Загрузка предварительно обученного классификатора для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Загрузка изображения
    image = cv2.imread(image)

    # Преобразование изображения в монохромное для улучшения производительности
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на изображении
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        face_features = extract_face_features(face_image)
    return face_features'''




'''
def get_gender(image):
    if image is None:
        print("Ошибка: Изображение лица не найдено.")
        return None
    # Загрузка предварительно обученной модели для классификации пола
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('/home/myrza/computer_vision/data/haarcascades/haarcascade_frontalface_default.xml')

    # Загрузка предварительно обученной модели для классификации пола
    gender_model = cv2.dnn.readNetFromCaffe('/home/myrza/computer_vision/face_recognize/deploy_gender.prototxt', '/home/myrza/computer_vision/face_recognize/gender_net.caffemodel')

    # Загрузка изображения
    image = cv2.imread(image)

    # Преобразование изображения в монохромное для улучшения производительности
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на изображении
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Извлечение области лица
        face_roi = gray[y:y+h, x:x+w]

        # Преобразование области лица в формат, необходимый для классификации гендера
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Применение модели для классификации гендера
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()

        # Получение метки класса с наибольшей вероятностью
        gender = 'Male' if gender_preds[0][0] > 0.5 else 'Female'

        return gender

def get_face(image):
    male_badpeople = [badpeople.feature for badpeople in Badpeople.objects.filter(sex='male')]
    female_badpeople = [badpeople.feature for badpeople in Badpeople.objects.filter(sex='female')]
    all_badpeople_features = [person.feature for person in Badpeople.objects.all()]
    # Инициализация детектора лиц и загрузка модели распознавания лиц
    detector = dlib.get_frontal_face_detector()
    # Загрузка изображения
    gender = get_gender(image)
    image = cv2.imread(image)

    # Преобразование изображения в монохромное для улучшения производительности
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на изображении с помощью детектора лиц из библиотеки dlib
    faces = detector(gray)
    for face in faces:
        # Извлечение изображения лица
        face_image = image[face.top():face.bottom(), face.left():face.right()]
        # Извлечение характеристик лица
        face_features = extract_face_features(face_image)
        if gender == 'Female':
            for badperson in female_badpeople:
                similarity = compare_features(badperson, face_features)
                print(similarity)
                if similarity > 0.09:
                    return ('рядом плохой человек')
                else:
                    print('хороший человек')
        elif gender == 'Male':
            for badperson in male_badpeople:
                similarity = compare_features(badperson, face_features)
                print(similarity)
                if similarity > 0.09:
                    return ('рядом плохой человек')
                else:
                    print('хороший человек')
        else:
            for badperson in all_badpeople_features:
                similarity = compare_features(badperson, face_features)
                print(similarity)
                if similarity > 0.09:
                    return('рядом плохой человек')
                else:
                    print('хороший человек')
        return ('произошла ошибка')'''