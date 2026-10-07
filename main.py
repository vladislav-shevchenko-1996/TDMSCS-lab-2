import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Вимикаємо логування TensorFlow
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------------------------------------------------
# 1. Створення синтетичних даних
# -------------------------------------------------------------------
def create_data(n_features, n_samples=1000):
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples)  # 3 класи

    # One-hot encoding
    y_onehot = np.zeros((n_samples, 3))
    y_onehot[np.arange(n_samples), y] = 1

    return tf.constant(X), tf.constant(y_onehot, dtype=tf.float32)

# -------------------------------------------------------------------
# 2. Функції побудови графіків
# -------------------------------------------------------------------
def plot_results(sizes, times_no_xla, times_xla, title_suffix=""):
    speedup = [no / xla if xla > 0 else 1.0 for no, xla in zip(times_no_xla, times_xla)]

    plt.figure(figsize=(12, 4.5))

    # Графік 1: Час виконання
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times_no_xla, 'o-', label='Без XLA', linewidth=2)
    plt.plot(sizes, times_xla, 's-', label='З XLA', linewidth=2)
    plt.xlabel('Кількість вхідних ознак')
    plt.ylabel('Час навчання (сек)')
    plt.title(f'Час навчання {title_suffix}')
    plt.legend()
    plt.grid(True)

    # Графік 2: Прискорення
    plt.subplot(1, 2, 2)
    plt.plot(sizes, speedup, 'o-', color='green', linewidth=2)
    plt.xlabel('Кількість вхідних ознак')
    plt.ylabel('Прискорення (x)')
    plt.title(f'Прискорення з XLA {title_suffix}')
    plt.axhline(y=1, color='red', linestyle='--', label='Без прискорення (1x)')
    plt.grid(True)
    plt.legend()

    for size, sp in zip(sizes, speedup):
        plt.annotate(f'{sp:.2f}x', (size, sp), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.show()

# ===================================================================
# БАЗОВИЙ ПРИКЛАД (Методичні вказівки)
# ===================================================================
@tf.function
def train_step_base_no_xla(X, y, w1, b1, w2, b2):
    with tf.GradientTape() as tape:
        h1 = tf.nn.relu(tf.matmul(X, w1) + b1)
        logits = tf.matmul(h1, w2) + b2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, logits))
    vars = [w1, b1, w2, b2]
    grads = tape.gradient(loss, vars)
    return loss, grads

@tf.function(jit_compile=True)
def train_step_base_xla(X, y, w1, b1, w2, b2):
    with tf.GradientTape() as tape:
        h1 = tf.nn.relu(tf.matmul(X, w1) + b1)
        logits = tf.matmul(h1, w2) + b2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, logits))
    vars = [w1, b1, w2, b2]
    grads = tape.gradient(loss, vars)
    return loss, grads

def run_baseline_benchmark(input_sizes, epochs=30):
    print("\n--- Базовий бенчмарк з методичних вказівок ---")
    times_no_xla, times_xla = [], []

    for size in input_sizes:
        X, y = create_data(size)
        hidden_size = max(10, size // 2)

        # Ініціалізація
        w1_no = tf.Variable(tf.random.normal([size, hidden_size], stddev=0.1))
        b1_no = tf.Variable(tf.zeros([hidden_size]))
        w2_no = tf.Variable(tf.random.normal([hidden_size, 3], stddev=0.1))
        b2_no = tf.Variable(tf.zeros([3]))
        opt_no = tf.optimizers.Adam(0.01)

        w1_xla = tf.Variable(w1_no.value())
        b1_xla = tf.Variable(b1_no.value())
        w2_xla = tf.Variable(w2_no.value())
        b2_xla = tf.Variable(b2_no.value())
        opt_xla = tf.optimizers.Adam(0.01)

        # Без XLA
        start = time.time()
        for _ in range(epochs):
            loss, grads = train_step_base_no_xla(X, y, w1_no, b1_no, w2_no, b2_no)
            opt_no.apply_gradients(zip(grads, [w1_no, b1_no, w2_no, b2_no]))
        t_no = time.time() - start

        # З XLA (з Warm-up)
        for _ in range(3):
            loss, grads = train_step_base_xla(X, y, w1_xla, b1_xla, w2_xla, b2_xla)
            opt_xla.apply_gradients(zip(grads, [w1_xla, b1_xla, w2_xla, b2_xla]))

        start = time.time()
        for _ in range(epochs):
            loss, grads = train_step_base_xla(X, y, w1_xla, b1_xla, w2_xla, b2_xla)
            opt_xla.apply_gradients(zip(grads, [w1_xla, b1_xla, w2_xla, b2_xla]))
        t_xla = time.time() - start

        print(f"Розмір входу: {size:<3} | Без XLA: {t_no:.4f} с | З XLA: {t_xla:.4f} с | Прискорення: {t_no/t_xla:.2f}x")
        times_no_xla.append(t_no)
        times_xla.append(t_xla)

    plot_results(input_sizes, times_no_xla, times_xla, "(Базовий приклад)")

# ===================================================================
# ДОДАТКОВЕ ЗАВДАННЯ 1: Заміна функції активації на Sigmoid
# ===================================================================
@tf.function
def train_step_sigmoid_no_xla(X, y, w1, b1, w2, b2):
    with tf.GradientTape() as tape:
        h1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1)  # Sigmoid замість ReLU
        logits = tf.matmul(h1, w2) + b2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, logits))
    vars = [w1, b1, w2, b2]
    grads = tape.gradient(loss, vars)
    return loss, grads

@tf.function(jit_compile=True)
def train_step_sigmoid_xla(X, y, w1, b1, w2, b2):
    with tf.GradientTape() as tape:
        h1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1)  # Sigmoid замість ReLU
        logits = tf.matmul(h1, w2) + b2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, logits))
    vars = [w1, b1, w2, b2]
    grads = tape.gradient(loss, vars)
    return loss, grads

def run_task_1(input_sizes, epochs=30):
    print("\n--- Завдання 1: Вплив функції активації (Sigmoid) ---")
    times_no_xla, times_xla = [], []

    for size in input_sizes:
        X, y = create_data(size)
        hidden_size = max(10, size // 2)

        w1_no = tf.Variable(tf.random.normal([size, hidden_size], stddev=0.1))
        b1_no = tf.Variable(tf.zeros([hidden_size]))
        w2_no = tf.Variable(tf.random.normal([hidden_size, 3], stddev=0.1))
        b2_no = tf.Variable(tf.zeros([3]))
        opt_no = tf.optimizers.Adam(0.01)

        w1_xla = tf.Variable(w1_no.value())
        b1_xla = tf.Variable(b1_no.value())
        w2_xla = tf.Variable(w2_no.value())
        b2_xla = tf.Variable(b2_no.value())
        opt_xla = tf.optimizers.Adam(0.01)

        # Без XLA
        start = time.time()
        for _ in range(epochs):
            loss, grads = train_step_sigmoid_no_xla(X, y, w1_no, b1_no, w2_no, b2_no)
            opt_no.apply_gradients(zip(grads, [w1_no, b1_no, w2_no, b2_no]))
        t_no = time.time() - start

        # З XLA (Warm-up)
        for _ in range(3):
            loss, grads = train_step_sigmoid_xla(X, y, w1_xla, b1_xla, w2_xla, b2_xla)
            opt_xla.apply_gradients(zip(grads, [w1_xla, b1_xla, w2_xla, b2_xla]))

        start = time.time()
        for _ in range(epochs):
            loss, grads = train_step_sigmoid_xla(X, y, w1_xla, b1_xla, w2_xla, b2_xla)
            opt_xla.apply_gradients(zip(grads, [w1_xla, b1_xla, w2_xla, b2_xla]))
        t_xla = time.time() - start

        print(f"Розмір входу: {size:<3} | Без XLA: {t_no:.4f} с | З XLA: {t_xla:.4f} с | Прискорення: {t_no/t_xla:.2f}x")
        times_no_xla.append(t_no)
        times_xla.append(t_xla)

    plot_results(input_sizes, times_no_xla, times_xla, "(Активація Sigmoid)")

# ===================================================================
# ТОЧКА ВХОДУ (ГОЛОВНИЙ ЦИКЛ)
# ===================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  ДОСЛІДЖЕННЯ ЕФЕКТИВНОСТІ XLA-КОМПІЛЯЦІЇ У TENSORFLOW (CPU)")
    print("=" * 65)

    input_sizes = [10, 50, 100, 200, 500]

    # Базовий бенчмарк з методичних вказівок
    run_baseline_benchmark(input_sizes, epochs=30)

    # Додаткові завдання
    run_task_1(input_sizes, epochs=30)             # Завдання 1: Sigmoid

    print("\n✅ Усі дослідження успішно виконано!")