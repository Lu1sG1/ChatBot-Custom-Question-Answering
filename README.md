# Custom Question Answering
Se implementa Simple Transformer para un chatbot que puede dar respuestas en base al contexto existente, de igual manera, se implementa Pytorch para realizar el entrenamiento mediante aceleración de GPU con CUDA

# Librerias 
![image](https://github.com/user-attachments/assets/2c30a544-3f17-45da-85d9-446d96288c9d)

* Torch - Aceleración mediante GPU
* Json - Lectura de archivos de entrenamiento y test
* Sklearn - Evalución y entrenamiento de modelo
* Simple Transformers, Question Answering Model - Modelo para implementar el sistema de pregunta y respuesta

# CUDA

Nos aseguramos que la aceleración por GPU esté disponible

![image](https://github.com/user-attachments/assets/40d3ac13-fdfc-474a-980b-6d63b5e3a522)



# Archivos de entrenamiento y test

Importamos los archivos de entrenamiento y test 
- Checar formato en train.json y test.json

![image](https://github.com/user-attachments/assets/827c5d2d-03d8-4e2d-b6f0-0f3dedd27799)
------------------------------------------------------------------
![image](https://github.com/user-attachments/assets/37bf9ba2-924d-4a56-9c3c-f79add1aea2e)


# Modelo 

Elegimos el modelo a usar, se soportan distintos tipos de modelos, por eficiencia se escoge la versión base de BERT, "bert-base-cased"

Después se ajustan los parámetros del modelo. 

![image](https://github.com/user-attachments/assets/bb0c1782-a95c-41a0-abdc-c4afce56f000)


# Entrenamiento 

Definimos el modelo, nos aseguramos de utilizar la aceleración por GPU y procedemos al entrenamieno 
![image](https://github.com/user-attachments/assets/ef53c43c-151f-4d55-85cf-619a3cf5c481)

Obtenemos los resultados
![image](https://github.com/user-attachments/assets/6d86001d-7201-4891-9434-52947805a385)

# Pruebas

Procedemos a las pruebas, donde le damos un contexto al modelo y hacemos una pregunta
![image](https://github.com/user-attachments/assets/63dc550f-f350-4ae0-b3a9-17de0019ab81)

![image](https://github.com/user-attachments/assets/a16e2e75-a077-442a-91b0-0e2b75a27f43)

![image](https://github.com/user-attachments/assets/d9f1652b-3a9b-4551-b1d5-c4a6964bf346)

![image](https://github.com/user-attachments/assets/e5d7d555-d732-44ce-b5cc-6b94607eca93)

# Métricas 

![image](https://github.com/user-attachments/assets/536e4455-b7a4-49b8-912a-b200646ef924)

![image](https://github.com/user-attachments/assets/3eaa7123-e823-40e3-a90f-0674f625aa78)

![image](https://github.com/user-attachments/assets/a9b7c35c-3e4d-45cb-91bb-a0c7477f8ab7)


