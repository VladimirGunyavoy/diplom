# Проверка расчетов линеаризации и дискретизации маятника

В этом документе представлены подробные расчёты для проверки корректности линеаризованных непрерывных матриц $A_c$ и $B_c$, а также формулы для дискретных матриц $A_d$ и $B_d$ на основе предоставленного кода Python.

## Теоретические основы

### Непрерывная динамика маятника

Система маятника описывается следующими нелинейными дифференциальными уравнениями состояния:

$$\dot{x} = f(x, u)$$

где $x = \begin{bmatrix} \theta \\ \dot{\theta} \end{bmatrix}$ - вектор состояния (угол и угловая скорость), а $u$ - управляющее воздействие.

Уравнения динамики, исходя из предоставленного кода, выглядят следующим образом:

$$\begin{cases}
\dot{\theta} = \dot{\theta} \\
\ddot{\theta} = -\frac{g}{l} \sin(\theta) - \text{damping} \cdot \dot{\theta} + u
\end{cases}$$

### Линеаризация

Линеаризация системы выполняется вокруг заданной точки состояния $x_0 = \begin{bmatrix} \theta_0 \\ \dot{\theta}_0 \end{bmatrix}$ и управляющего воздействия $u_0 = 0$. Линеаризованная система в непрерывном времени имеет вид:

$$\dot{x} = A_c x + B_c u$$

Матрицы $A_c$ и $B_c$ вычисляются как якобианы функции $f(x,u)$ по $x$ и $u$ соответственно, в точке линеаризации:

$$A_c = \frac{\partial f}{\partial x}\bigg|_{x_0, u_0} = \begin{bmatrix}
\frac{\partial \dot{\theta}}{\partial \theta} & \frac{\partial \dot{\theta}}{\partial \dot{\theta}} \\
\frac{\partial \ddot{\theta}}{\partial \theta} & \frac{\partial \ddot{\theta}}{\partial \dot{\theta}}
\end{bmatrix}\bigg|_{x_0} = \begin{bmatrix}
0 & 1 \\
-\frac{g}{l} \cos(\theta_0) & -\text{damping}
\end{bmatrix}$$

$$B_c = \frac{\partial f}{\partial u}\bigg|_{x_0, u_0} = \begin{bmatrix}
\frac{\partial \dot{\theta}}{\partial u} \\
\frac{\partial \ddot{\theta}}{\partial u}
\end{bmatrix}\bigg|_{u_0} = \begin{bmatrix}
0 \\
1
\end{bmatrix}$$

**Используемые параметры из кода:**
* $g = 9.81$ м/с$^2$
* $l = 2.0$ м
* $\text{damping} = 0.1$

Подставляя значения $g$ и $l$, получаем:

$$A_c = \begin{bmatrix}
0 & 1 \\
-\frac{9.81}{2.0} \cos(\theta_0) & -0.1
\end{bmatrix} = \begin{bmatrix}
0 & 1 \\
-4.905 \cos(\theta_0) & -0.1
\end{bmatrix}$$

$$B_c = \begin{bmatrix}
0 \\
1
\end{bmatrix}$$

### Дискретизация и шаг по времени

Дискретизация непрерывной системы с использованием матричной экспоненты для шага по времени $dt$:

$$\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} = \exp\left( \begin{bmatrix} A_c & B_c \\ 0 & 0 \end{bmatrix} \cdot dt \right)$$

где $I$ - единичная матрица соответствующего размера.

После вычисления матриц $A_d$ и $B_d$, следующее состояние $x_{k+1}$ системы в дискретном времени определяется по формуле:

$$x_{k+1} = A_d x_k + B_d u_k$$

где $x_k$ - текущее состояние, а $u_k$ - управляющее воздействие на текущем шаге.

## Расчёты для заданных точек состояния

Для каждой точки состояния $( \theta_0, \dot{\theta}_0 )$ будут вычислены линеаризованные непрерывные матрицы $A_c$ и $B_c$. Матрица $B_c$ является константой и не зависит от состояния. Затем будут представлены формулы для дискретных матриц $A_d$ и $B_d$ и следующего состояния $x_{k+1}$.

Для получения численных значений дискретных матриц и следующего состояния, вам необходимо запустить код, используя функцию `discretize` с выбранным $dt$ и `discrete_step` с выбранным $dt$ и $u$. Для демонстрации мы будем использовать $dt = 0.1$ и $u = 0.0$.

---

### 1. Точка $( \theta_0, \dot{\theta}_0 ) = (0, 0)$

**Линеаризованные непрерывные матрицы:**

$$A_c = \begin{bmatrix}
0 & 1 \\
-4.905 \cos(0) & -0.1
\end{bmatrix} = \begin{bmatrix}
0 & 1 \\
-4.905 \cdot 1 & -0.1
\end{bmatrix} = \begin{bmatrix}
0 & 1 \\
-4.905 & -0.1
\end{bmatrix}$$

$$B_c = \begin{bmatrix}
0 \\
1
\end{bmatrix}$$

**Расчёт дискретных матриц ($A_d, B_d$):**

Для $dt = 0.1$, расширенная матрица для экспоненты:

$$M \cdot dt = \begin{bmatrix}
A_c & B_c \\
0 & 0
\end{bmatrix} \cdot 0.1 = \begin{bmatrix}
0 & 1 & 0 \\
-4.905 & -0.1 & 1 \\
0 & 0 & 0
\end{bmatrix} \cdot 0.1 = \begin{bmatrix}
0 & 0.1 & 0 \\
-0.4905 & -0.01 & 0.1 \\
0 & 0 & 0
\end{bmatrix}$$

Дискретные матрицы $A_d$ и $B_d$ получаются из:

$$\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} = \exp\left( \begin{bmatrix} 0 & 0.1 & 0 \\ -0.4905 & -0.01 & 0.1 \\ 0 & 0 & 0 \end{bmatrix} \right)$$

**Следующее состояние $x_{k+1}$ при $x_k = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ и $u_k = 0.0$:**

$$x_{k+1} = A_d \begin{bmatrix} 0 \\ 0 \end{bmatrix} + B_d \cdot 0.0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

*Численные значения $A_d, B_d$ и $x_{k+1}$ будут получены при выполнении метода `discretize` и `discrete_step` в вашем коде.*

---

### 2. Точка $( \theta_0, \dot{\theta}_0 ) = (1, 0)$

**Линеаризованные непрерывные матрицы (угол $\theta_0 = 1$ радиан):**

$$A_c = \begin{bmatrix}
0 & 1 \\
-4.905 \cos(1) & -0.1
\end{bmatrix} \approx \begin{bmatrix}
0 & 1 \\
-2.6503 & -0.1
\end{bmatrix}$$

$$B_c = \begin{bmatrix}
0 \\
1
\end{bmatrix}$$

**Расчёт дискретных матриц ($A_d, B_d$):**

Для $dt = 0.1$, расширенная матрица для экспоненты:

$$M \cdot dt = \begin{bmatrix}
A_c & B_c \\
0 & 0
\end{bmatrix} \cdot 0.1 \approx \begin{bmatrix}
0 & 0.1 & 0 \\
-0.26503 & -0.01 & 0.1 \\
0 & 0 & 0
\end{bmatrix}$$

Дискретные матрицы $A_d$ и $B_d$ получаются из:

$$\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} = \exp\left( \begin{bmatrix} 0 & 0.1 & 0 \\ -0.26503 & -0.01 & 0.1 \\ 0 & 0 & 0 \end{bmatrix} \right)$$

**Следующее состояние $x_{k+1}$ при $x_k = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ и $u_k = 0.0$:**

$$x_{k+1} = A_d \begin{bmatrix} 1 \\ 0 \end{bmatrix} + B_d \cdot 0.0$$

*Численные значения $A_d, B_d$ и $x_{k+1}$ будут получены при выполнении метода `discretize` и `discrete_step` в вашем коде.*

---

### 3. Точка $( \theta_0, \dot{\theta}_0 ) = (0, 1)$

**Линеаризованные непрерывные матрицы:**

$$A_c = \begin{bmatrix}
0 & 1 \\
-4.905 \cos(0) & -0.1
\end{bmatrix} = \begin{bmatrix}
0 & 1 \\
-4.905 \cdot 1 & -0.1
\end{bmatrix} = \begin{bmatrix}
0 & 1 \\
-4.905 & -0.1
\end{bmatrix}$$

$$B_c = \begin{bmatrix}
0 \\
1
\end{bmatrix}$$

**Расчёт дискретных матриц ($A_d, B_d$):**

Для $dt = 0.1$, расширенная матрица для экспоненты та же, что и для точки $(0, 0)$:

$$M \cdot dt = \begin{bmatrix}
0 & 0.1 & 0 \\
-0.4905 & -0.01 & 0.1 \\
0 & 0 & 0
\end{bmatrix}$$

Дискретные матрицы $A_d$ и $B_d$ получаются из:

$$\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} = \exp\left( \begin{bmatrix} 0 & 0.1 & 0 \\ -0.4905 & -0.01 & 0.1 \\ 0 & 0 & 0 \end{bmatrix} \right)$$

**Следующее состояние $x_{k+1}$ при $x_k = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ и $u_k = 0.0$:**

$$x_{k+1} = A_d \begin{bmatrix} 0 \\ 1 \end{bmatrix} + B_d \cdot 0.0$$

*Численные значения $A_d, B_d$ и $x_{k+1}$ будут получены при выполнении метода `discretize` и `discrete_step` в вашем коде.*

---

### 4. Точка $( \theta_0, \dot{\theta}_0 ) = (1, 1)$

**Линеаризованные непрерывные матрицы (угол $\theta_0 = 1$ радиан):**

$$A_c = \begin{bmatrix}
0 & 1 \\
-4.905 \cos(1) & -0.1
\end{bmatrix} \approx \begin{bmatrix}
0 & 1 \\
-2.6503 & -0.1
\end{bmatrix}$$

$$B_c = \begin{bmatrix}
0 \\
1
\end{bmatrix}$$

**Расчёт дискретных матриц ($A_d, B_d$):**

Для $dt = 0.1$, расширенная матрица для экспоненты та же, что и для точки $(1, 0)$:

$$M \cdot dt = \begin{bmatrix}
0 & 0.1 & 0 \\
-0.26503 & -0.01 & 0.1 \\
0 & 0 & 0
\end{bmatrix}$$

Дискретные матрицы $A_d$ и $B_d$ получаются из:

$$\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} = \exp\left( \begin{bmatrix} 0 & 0.1 & 0 \\ -0.26503 & -0.01 & 0.1 \\ 0 & 0 & 0 \end{bmatrix} \right)$$

**Следующее состояние $x_{k+1}$ при $x_k = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ и $u_k = 0.0$:**

$$x_{k+1} = A_d \begin{bmatrix} 1 \\ 1 \end{bmatrix} + B_d \cdot 0.0$$

*Численные значения $A_d, B_d$ и $x_{k+1}$ будут получены при выполнении метода `discretize` и `discrete_step` в вашем коде.*

---

### 5. Точка $( \theta_0, \dot{\theta}_0 ) = (\pi/4, 0)$

**Линеаризованные непрерывные матрицы:**

$$A_c = \begin{bmatrix}
0 & 1 \\
-4.905 \cos(\pi/4) & -0.1
\end{bmatrix} \approx \begin{bmatrix}
0 & 1 \\
-3.468 & -0.1
\end{bmatrix}$$

$$B_c = \begin{bmatrix}
0 \\
1
\end{bmatrix}$$

**Расчёт дискретных матриц ($A_d, B_d$):**

Для $dt = 0.1$, расширенная матрица для экспоненты:

$$M \cdot dt = \begin{bmatrix}
A_c & B_c \\
0 & 0
\end{bmatrix} \cdot 0.1 \approx \begin{bmatrix}
0 & 0.1 & 0 \\
-0.3468 & -0.01 & 0.1 \\
0 & 0 & 0
\end{bmatrix}$$

Дискретные матрицы $A_d$ и $B_d$ получаются из:

$$\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} = \exp\left( \begin{bmatrix} 0 & 0.1 & 0 \\ -0.3468 & -0.01 & 0.1 \\ 0 & 0 & 0 \end{bmatrix} \right)$$

**Следующее состояние $x_{k+1}$ при $x_k = \begin{bmatrix} \pi/4 \\ 0 \end{bmatrix}$ и $u_k = 0.0$:**

$$x_{k+1} = A_d \begin{bmatrix} \pi/4 \\ 0 \end{bmatrix} + B_d \cdot 0.0$$

*Численные значения $A_d, B_d$ и $x_{k+1}$ будут получены при выполнении метода `discretize` и `discrete_step` в вашем коде.*

---

### 6. Точка $( \theta_0, \dot{\theta}_0 ) = (\pi/2, 0)$

**Линеаризованные непрерывные матрицы:**

$$A_c = \begin{bmatrix}
0 & 1 \\
-4.905 \cos(\pi/2) & -0.1
\end{bmatrix} = \begin{bmatrix}
0 & 1 \\
0 & -0.1
\end{bmatrix}$$

$$B_c = \begin{bmatrix}
0 \\
1
\end{bmatrix}$$

**Расчёт дискретных матриц ($A_d, B_d$):**

Для $dt = 0.1$, расширенная матрица для экспоненты:

$$M \cdot dt = \begin{bmatrix}
A_c & B_c \\
0 & 0
\end{bmatrix} \cdot 0.1 = \begin{bmatrix}
0 & 0.1 & 0 \\
0 & -0.01 & 0.1 \\
0 & 0 & 0
\end{bmatrix}$$

Дискретные матрицы $A_d$ и $B_d$ получаются из:

$$\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} = \exp\left( \begin{bmatrix} 0 & 0.1 & 0 \\ 0 & -0.01 & 0.1 \\ 0 & 0 & 0 \end{bmatrix} \right)$$

**Следующее состояние $x_{k+1}$ при $x_k = \begin{bmatrix} \pi/2 \\ 0 \end{bmatrix}$ и $u_k = 0.0$:**

$$x_{k+1} = A_d \begin{bmatrix} \pi/2 \\ 0 \end{bmatrix} + B_d \cdot 0.0$$

*Численные значения $A_d, B_d$ и $x_{k+1}$ будут получены при выполнении метода `discretize` и `discrete_step` в вашем коде.*

---

### 7. Точка $( \theta_0, \dot{\theta}_0 ) = (\pi, 0)$

**Линеаризованные непрерывные матрицы:**

$$A_c = \begin{bmatrix}
0 & 1 \\
-4.905 \cos(\pi) & -0.1
\end{bmatrix} = \begin{bmatrix}
0 & 1 \\
-4.905 \cdot (-1) & -0.1
\end{bmatrix} = \begin{bmatrix}
0 & 1 \\
4.905 & -0.1
\end{bmatrix}$$

$$B_c = \begin{bmatrix}
0 \\
1
\end{bmatrix}$$

**Расчёт дискретных матриц ($A_d, B_d$):**

Для $dt = 0.1$, расширенная матрица для экспоненты:

$$M \cdot dt = \begin{bmatrix}
A_c & B_c \\
0 & 0
\end{bmatrix} \cdot 0.1 = \begin{bmatrix}
0 & 0.1 & 0 \\
0.4905 & -0.01 & 0.1 \\
0 & 0 & 0
\end{bmatrix}$$

Дискретные матрицы $A_d$ и $B_d$ получаются из:

$$\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} = \exp\left( \begin{bmatrix} 0 & 0.1 & 0 \\ 0.4905 & -0.01 & 0.1 \\ 0 & 0 & 0 \end{bmatrix} \right)$$

**Следующее состояние $x_{k+1}$ при $x_k = \begin{bmatrix} \pi \\ 0 \end{bmatrix}$ и $u_k = 0.0$:**

$$x_{k+1} = A_d \begin{bmatrix} \pi \\ 0 \end{bmatrix} + B_d \cdot 0.0$$

*Численные значения $A_d, B_d$ и $x_{k+1}$ будут получены при выполнении метода `discretize` и `discrete_step` в вашем коде.*