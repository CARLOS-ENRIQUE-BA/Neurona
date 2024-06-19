import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Funciones principales

#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))

#def step(x):
#    return np.where(x >= 0, 1, 0)

def predecir(X, w, b):
    return np.dot(X, w) + b #Se multiplica la matris de las x con la de las w, al resultado se le suma el sesgo

#def predecir(X, w, b):
#    z = np.dot(X, w) + b  # Salida lineal
#    return sigmoid(z)  # Aplicar función sigmoide a la salida lineal

def error_cuadratico_medio(y_deseada, y_calculada):
    return np.mean((y_deseada - y_calculada)**2) #Calcular el promedio del error mediante ECM, evaluando la diferencia de las y's y se eleva al cuadrado,
                                                 #con el fin de reducir los errores grandes y evitar los errores negativos

def entrenar(X, y, w, b, tasa_aprendizaje, epocas):
    m = X.shape[0] #Guardamos el numero de filas de X
    
    historia_pesos = np.zeros((epocas, w.shape[0]))
    historia_sesgos = np.zeros(epocas)
    historia_costos = np.zeros(epocas)
    historia_predicciones = []

    for epoca in range(epocas):
        y_calculada = predecir(X, w, b) #Calcula la y_calculada con base a las X, los pesoss y el sesgo
        
        error = y_calculada - y
        
        dw = (2/m) * np.dot(X.T, error) #Calcula la gradiente de los pesos, con base a un vector que indica cómo deben cambiar los pesos para reducir la función de pérdida
        db = (2/m) * np.sum(error)
        
        #   Para una funcion de activacion
        #dw = (2/m) * np.dot(X.T, error * y_calculada * (1 - y_calculada))  # Gradiente de los pesos con función de activación sigmoidal
        #db = (2/m) * np.sum(error * y_calculada * (1 - y_calculada))
        
        w -= tasa_aprendizaje * dw #Actualiza los pesos del modelo moviéndolos en la dirección opuesta al gradiente. Controla el tamaño del paso
        b -= tasa_aprendizaje * db #
        
        costo = error_cuadratico_medio(y, y_calculada)
        historia_costos[epoca] = costo
        historia_pesos[epoca, :] = w
        historia_sesgos[epoca] = b
        historia_predicciones.append(y_calculada)
        
    
    return w, b, historia_pesos, historia_sesgos, historia_costos, historia_predicciones

# Configuración inicial de datos
conjunto_datos = pd.read_excel('data/221188.xlsx')

x1 = conjunto_datos['x1'].values
x2 = conjunto_datos['x2'].values
x3 = conjunto_datos['x3'].values
x4 = conjunto_datos['x4'].values
x5 = conjunto_datos['x5'].values
x6 = conjunto_datos['x6'].values
yd = conjunto_datos['y'].values

x = np.column_stack((x1, x2, x3, x4, x5, x6))
y_deseada = yd

#escalador = StandardScaler()
#x = escalador.fit_transform(x)

# Valores iniciales aleatorios
np.random.seed(0)  # Semilla para reproducibilidad
#inicializacion aleatoria de los datos
w_inicial = np.random.rand(6)
b_inicial = np.random.rand()

def graficar_diferencias(y_deseada, y_predichas, epocas):
    plt.figure(figsize=(15, 10))
    epocas_a_mostrar = [0, epocas // 2, epocas - 1]
    for i, epoca in enumerate(epocas_a_mostrar):
        plt.subplot(3, 1, i + 1)
        plt.plot(y_deseada, label='Y Deseada')
        plt.plot(y_predichas[epoca], label='Y Predicha')
        plt.title(f'Epoca {epoca + 1}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Función para graficar el error
def graficar_error(historia_costos):
    plt.figure(figsize=(10, 6))
    plt.plot(historia_costos, label='Error Cuadrático Medio')
    plt.xlabel('Época')
    plt.ylabel('ECM')
    plt.title('Evolución del Error Cuadrático Medio')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para entrenar y actualizar la interfaz
def entrenar_y_actualizar():
    global w_inicial, b_inicial
    
    w_inicial = np.random.rand(6)  # Actualizamos w_inicial con valores aleatorios
    b_inicial = np.random.rand()   # Actualizamos b_inicial con valor aleatorio
    
    w = w_inicial.copy()
    b = b_inicial

    tasa_aprendizaje = float(tasa_aprendizaje_entry.get()) #Tasa de Aprendizaje determina el tamaño de los pasos que el algoritmo de entrenamiento da al ajustar los pesos y sesgos del modelo
    epocas = int(epocas_entry.get())
    
    w, b, historia_pesos, historia_sesgos, historia_costos, historia_predicciones = entrenar(x, y_deseada, w, b, tasa_aprendizaje, epocas)
    
    y_calculada = predecir(x, w, b) #calcular la prediccion del modelo
    costo_final = error_cuadratico_medio(y_deseada, y_calculada) #saca la ECM. Nos sirve para evaluar el rendimiento del modelo después de completar el proceso de entrenamiento.
    
    
    # Actualizar tabla de pesos y sesgo
    for i in range(len(w)):
        tabla.set(f"final_{i}", 1, f"{w_inicial[i]:.4f}")  
        tabla.set(f"final_{i}", 2, f"{w[i]:.4f}")
    
    tabla.set("final_sesgo", 1, f"{b_inicial:.4f}")  
    tabla.set("final_sesgo", 2, f"{b:.4f}")
    
    # Limpiar figuras de matplotlib
    plt.close('all')
    
    # Gráfico de evolución de pesos y sesgo
    plt.figure(figsize=(10, 6))
    for i in range(w.shape[0]):
        plt.plot(historia_pesos[:, i], label=f'Peso {i+1}')
    plt.plot(historia_sesgos, label='Sesgo', linestyle='--')
    plt.xlabel('Época')
    plt.ylabel('Valor')
    plt.title('Evolución de Pesos y Sesgo')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Gráfico de diferencias
    graficar_diferencias(y_deseada, historia_predicciones, epocas)
    
    # Gráfico de error
    graficar_error(historia_costos)

# Interfaz gráfica (GUI)
ventana = tk.Tk()
ventana.title('Entrenamiento del Modelo')

# Estilos
style = ttk.Style()
style.configure('TLabel', font=('Helvetica', 12))
style.configure('TButton', font=('Helvetica', 12), padding=10)
style.configure('TEntry', font=('Helvetica', 12))

# Marco principal
mainframe = ttk.Frame(ventana, padding="20 20 20 20")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Etiquetas y campos de entrada
ttk.Label(mainframe, text="Tasa de Aprendizaje:").grid(column=0, row=0, padx=10, pady=5)
tasa_aprendizaje_entry = ttk.Entry(mainframe)
tasa_aprendizaje_entry.grid(column=1, row=0, padx=10, pady=5)
tasa_aprendizaje_entry.insert(tk.END, '0.1')

ttk.Label(mainframe, text="Épocas:").grid(column=0, row=1, padx=10, pady=5)
epocas_entry = ttk.Entry(mainframe)
epocas_entry.grid(column=1, row=1, padx=10, pady=5)
epocas_entry.insert(tk.END, '1000')

# Botón para iniciar entrenamiento
entrenar_button = ttk.Button(mainframe, text="Entrenar", command=entrenar_y_actualizar)
entrenar_button.grid(column=0, row=2, columnspan=2, pady=10)

# Etiquetas para mostrar resultados
costo_label = ttk.Label(mainframe, text="")
costo_label.grid(column=0, row=3, columnspan=2, pady=5)

costo_final_label = ttk.Label(mainframe, text="")
costo_final_label.grid(column=0, row=4, columnspan=2, pady=5)

# Tabla de pesos y sesgo
columns = ('Característica', 'Inicial', 'Final')
tabla = ttk.Treeview(mainframe, columns=columns, show='headings')

for col in columns:
    tabla.heading(col, text=col)
    tabla.column(col, anchor=tk.CENTER)

pesos_y_sesgo = [('w1', '0', '0'), ('w2', '0', '0'), ('w3', '0', '0'), ('w4', '0', '0'), ('w5', '0', '0'), ('w6', '0', '0'), ('Sesgo', '0', '0')]
for idw, (caracteristica, inicial, final) in enumerate(pesos_y_sesgo):
    item_id = f"final_{idw}" if caracteristica != 'Sesgo' else "final_sesgo"
    tabla.insert("", "end", iid=item_id, values=(caracteristica, inicial, final))

tabla.grid(column=0, row=5, columnspan=2, pady=10)

# Configurar la expansión de las celdas
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

ventana.mainloop()
