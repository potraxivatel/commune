# Pyramid: Training the Memory Writer (Qwen 2.5 0.5B -> GGUF)

Это инструкция по обучению первой специализированной модели для Pyramid на **Google Colab** (бесплатная T4 GPU) или **Kaggle**.

Обучение занимает всего **~5–10 минут**. На выходе получается готовый файл `.gguf` размером **~350 МБ**.

---

## Вариант 1: Через Google Colab (самый простой)

1. Открой [Google Colab](https://colab.research.google.com/).
2. Выбери вкладку **Upload** (Загрузить) и загрузи файл:
   [`training/Pyramid_Train_Memory_Writer.ipynb`](Pyramid_Train_Memory_Writer.ipynb)
3. В меню Colab выбери: **Runtime -> Change runtime type -> T4 GPU** (Среда выполнения -> Сменить тип -> T4 GPU).
4. Нажми **Runtime -> Run all** (Среда выполнения -> Выполнить всё).

Notebook сам:
* Установит сверхбыструю библиотеку **Unsloth**
* Загрузит базовую **Qwen 2.5 0.5B Instruct**
* Сгенерирует датасет
* Дообучит LoRA-адаптер на задачу `Memory Writer`
* Проверит инференс (тестовый запрос)
* Экспортирует и упакует модель прямо в **GGUF (Q4_K_M)**!
* Скачает файл `pyramid_memory_writer-unsloth.Q4_K_M.gguf` тебе на компьютер.

---

## Вариант 2: Подключение обученной модели в Pyramid

Когда файл `.gguf` скачан:
1. Перемести его в папку:
   `/home/void/Projects/Pyramid/llms/custom/pyramid_memory_writer.Q4_K_M.gguf`

2. Запусти тестовый скрипт проверки:
   ```bash
   python tests/test_custom_model.py
   ```

3. Теперь модель официально работает как специализированный компонент `Memory Writer` внутри Pyramid Engine!
