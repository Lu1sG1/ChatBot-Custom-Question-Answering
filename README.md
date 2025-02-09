# Custom Question Answering
Se implementa Simple Transformer para un chatbot que puede dar respuestas en base al contexto existente, de igual manera, se implementa Pytorch para realizar el entrenamiento mediante aceleración de GPU con CUDA

# Librerias 
![image](https://github.com/user-attachments/assets/2c30a544-3f17-45da-85d9-446d96288c9d)

* Torch - Aceleración mediante GPU
* Json - Lectura de archivos de entrenamiento y test
* Sklearn - Evalución y entrenamiento de modelo
* Simple Transformers, Question Answering Model - Modelo para implementar el sistema de pregunta y respuesta

# CUDA
![image](https://github.com/user-attachments/assets/40d3ac13-fdfc-474a-980b-6d63b5e3a522)

Nos aseguramos que la aceleración por GPU esté disponible

# Archivos de entrenamiento y test
![image](https://github.com/user-attachments/assets/827c5d2d-03d8-4e2d-b6f0-0f3dedd27799)
------------------------------------------------------------------
![image](https://github.com/user-attachments/assets/37bf9ba2-924d-4a56-9c3c-f79add1aea2e)
Importamos los archivos de entrenamiento y test 
- Checar formato en train.json y test.json

# Modelo 
![image](https://github.com/user-attachments/assets/bb0c1782-a95c-41a0-abdc-c4afce56f000)
Elegimos el modelo a usar, se soportan distintos tipos de modelos, por eficiencia se escoge la versión base de BERT, "bert-base-cased"

Después se ajustan los parámetros del modelo, a continuación se explica de manera breve el impacto de los parámetros. 

*reprocess_input_data: True
Qué hace: Si se establece en True, los datos de entrada se volverán a procesar incluso si ya existen características en caché.
Impacto: Garantiza que se apliquen los cambios en el preprocesamiento de datos, pero puede aumentar el tiempo de entrenamiento si el conjunto de datos es grande.

* overwrite_output_dir: True
  
Qué hace: Si se establece en True, el directorio de salida será sobrescrito si ya existe.

Impacto: Útil para evitar errores al volver a ejecutar experimentos, pero hay que tener precaución, ya que elimina los resultados anteriores.

* use_cached_eval_features: True
  
Qué hace: Si se establece en True, se usarán características de evaluación en caché para acelerar la evaluación.

Impacto: Reduce el tiempo de evaluación, especialmente para conjuntos de datos grandes, pero puede llevar a usar características desactualizadas si los datos cambian.

* output_dir: f"QA_project/outputs/{model_type}"
  
Qué hace: Especifica el directorio donde se guardarán los resultados del entrenamiento (p. ej., puntos de control del modelo, registros).

Impacto: Organiza los resultados por tipo de modelo, facilitando la gestión de múltiples experimentos.

* best_model_dir: f"QA_project/outputs/{model_type}/best_model"
  
Qué hace: Especifica el directorio donde se guardará el mejor modelo (según las métricas de evaluación).

Impacto: Permite un acceso fácil al modelo con mejor rendimiento para inferencia o ajuste adicional.

* evaluate_during_training: True
  
Qué hace: Si se establece en True, el modelo se evaluará en el conjunto de validación durante el entrenamiento.

Impacto: Proporciona retroalimentación en tiempo real sobre el rendimiento del modelo, ayudando a detectar sobreajuste o infraajuste.

* max_seq_length: 128
  
Qué hace: Establece la longitud máxima de la secuencia (en tokens) para el texto de entrada. Las secuencias más largas se truncarán y las más cortas se rellenarán.

Impacto: Afecta el uso de memoria y el rendimiento del modelo. Secuencias más largas capturan más contexto, pero requieren más memoria y cómputo.

* num_train_epochs: 50
  
Qué hace: Especifica el número de veces que el modelo recorrerá todo el conjunto de datos de entrenamiento.

Impacto: Más épocas pueden mejorar el rendimiento, pero pueden causar sobreajuste si el modelo es demasiado complejo o el conjunto de datos es pequeño.

* evaluate_during_training_steps: 1000
  
Qué hace: Especifica la frecuencia (en pasos) con la que el modelo se evaluará durante el entrenamiento.

Impacto: Evaluaciones frecuentes proporcionan retroalimentación más detallada, pero aumentan el tiempo de entrenamiento.

* wandb_project: "Question Answer Application"
  
Qué hace: Especifica el nombre del proyecto para registrar experimentos en Weights & Biases (W&B).

Impacto: Ayuda a organizar y rastrear experimentos en el panel de control de W&B.

* wandb_kwargs: {"name": model_name}
  
Qué hace: Especifica argumentos adicionales para el registro en W&B, como el nombre del experimento.

Impacto: Facilita la identificación y comparación de diferentes ejecuciones en el panel de W&B.

* save_model_every_epoch: False
  
Qué hace: Si se establece en True, el modelo se guardará al final de cada época.

Impacto: Ahorra espacio en disco al no guardar modelos intermedios, pero se pierde la capacidad de reanudar el entrenamiento desde cualquier época.

* save_eval_checkpoints: False
  
Qué hace: Si se establece en True, el modelo se guardará después de cada evaluación.

Impacto: Ahorra espacio en disco, pero limita la capacidad de recuperar el mejor modelo durante el entrenamiento.

* n_best_size: 3
  
Qué hace: Especifica el número de mejores predicciones a considerar al evaluar modelos de preguntas y respuestas.

Impacto: Afecta las métricas de evaluación (p. ej., F1-score, Exact Match). Un valor mayor considera más predicciones, pero puede aumentar el tiempo de cómputo.

* use_early_stopping: True
  
Qué hace: Si se establece en True, el entrenamiento se detendrá temprano si la métrica de evaluación deja de mejorar.

Impacto: Previene el sobreajuste y ahorra tiempo al detener el entrenamiento cuando es poco probable que haya más mejoras.

* n_gpu: 0
  
Qué hace: Especifica el número de GPUs a usar para el entrenamiento. Se establece en 0 para entrenar solo con CPU.

Impacto: Usar GPUs (n_gpu > 0) acelera significativamente el entrenamiento, especialmente para modelos y conjuntos de datos grandes.

* train_batch_size: 128
  
Qué hace: Especifica el número de ejemplos de entrenamiento procesados en un solo paso de forward/backward.

Impacto: Tamaños de lote más grandes aceleran el entrenamiento pero requieren más memoria. Tamaños más pequeños pueden mejorar la generalización, pero ralentizan el entrenamiento.

* eval_batch_size: 64
  
Qué hace: Especifica el número de ejemplos de evaluación procesados en un solo paso de forward.

Impacto: Tamaños de lote más grandes aceleran la evaluación, pero requieren más memoria. Tamaños más pequeños son útiles en entornos con restricciones de memoria.

* Parámetros Opcionales/Comentados

* early_stopping_metric: "mcc"
  
Qué hace: Especifica la métrica a monitorear para la detención temprana (p. ej., coeficiente de correlación de Matthews).

Impacto: Ayuda a detener el entrenamiento cuando la métrica elegida deja de mejorar.

* manual_seed: 4
Qué hace: Establece una semilla aleatoria para la reproducibilidad.

Impacto: Garantiza que los resultados sean consistentes entre ejecuciones.

* use_multiprocessing: False
Qué hace: Si se establece en True, se usará multiprocesamiento para la carga de datos.

Impacto: Acelera la carga de datos, pero puede aumentar el uso de memoria.

* config: {"output_hidden_states": True}
Qué hace: Especifica opciones adicionales de configuración del modelo, como la salida de estados ocultos.

Impacto: Útil para tareas avanzadas como extracción de características o análisis del modelo.

# Entrenamiento 

Definimos el modelo, nos aseguramos de utilizar la aceleración por GPU y procedemos al entrenamieno 
![image](https://github.com/user-attachments/assets/ef53c43c-151f-4d55-85cf-619a3cf5c481)


