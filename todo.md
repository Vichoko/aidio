* Opgimizar el forward del gmm para que no de segfault
    * Haciendo un flatten del batch para evitar for
    * Ver si se puede hacer en paralelo
* evaluar mfcc de ptorch
* Insights reunion jorge 23-4-2020
    * Darle mas bueltya los parametros
    * Probar WAvenet + Otros pooling
    * Disminuir el hidden size de los sentence encoder
        * La vecindad temporal es evidente, pero no la vecindad de features no tanto
        * 
    * Probar BiLSTM con otro Pooling tambien.
        * El profe le tiene más fe al max pooling 
    * Explicar modelos en mucho detalle, ojala con forulas
    * Enviar introduccion en latex
    * Usar Lefraru
   
* Parametrizar wavenet para meter mas batchsize
    * resnetx

            * S/N   svs_svd_2_0 val_a=0.869 test_a=0.85%

0    * wavenet
 0   python aidio.py model --model wavenet --experiment svs_svd_4 --data_path /home/voyanedel/data/data/1d/svs-svd-bin-full --label_filename labels.csv --gpus [0]
        * 1-1-48    svs_svd_0   val_a=0.607@81  #70k
        * 2-2-31    svs_svd_1   val_a=0.796@306 #103k
        * 3-2-24    svs_svd_2   val_a=0.847@297 #125k
        * 3-3-18    svs_svd_3   val_a=0.814  #159k
        * 4-3-14    svs_svd_4   val_a=0.858@  #193k
        
        Observaciones: 
            * Se demora entre 20 y 24 horas en converger a ~85%
    * GMM
        * val_a=0.725 (svs)
        * val_a=0.9000 (svs-svd)    test_a=0.8785
        * svs_svd_8_0   val_a=0.784
        
    * Wavenet Tranwformer
        * 1-1 8-1-36    svs_svd_1   val_a=0.45@22  #
        * 3-2 4-2-128-22    svs_svd_3   val_a=0.48@27
        (w/o positional encoding)
        * 2-2 2-1-128-20    svs_svd_4   val_a=0.55@75
        * 3-2 1-1-512-18    svs_svd_6   val_a=0.547@38
        * 3-2 2-1-512-16    svd_svd_7   val_a=0.81@58
        * 3-2 3-1-512-16    svd_svd_7   val_a=0.81@58
        * 4-3 3-3-512-13-0.01 (32 32)  svs_svd_2_1 
        * 4-2 3-2-256-6 (32 32) svs_svd_2_1 val_a=0.48
        * 4-3 4-1-256-  svs_svd_2_5 val_a=0.48


        
    * wavenet lstm
        * 1-1 128-1-36  svs_svd_3   val_a=0.75@125
        * 2-2 128-1-20  svs_svd_4   val_a=0.73@12   
        * 2-2 128-2-23  svs_svd_5   val_a=0.841@96;0.859@101
        * 3-2 512-1-15 (16 16)  svs_svd_6   val_a=0.862@59
        * 3-2 1024-1-16 (32 32) svs_svd_7   val_a=0.86@69   test_a=0.851
        * 4-3 256-2-16  (32 32) svs_svd_2_1 val_a=0.862 test_a=0.867**
        * 4-3 256-4-13  (16 16) svs_svd_2_2 val_a=0.853 test_a=0.839
        * 5-5 256-3-6   (16 16) svs_svd_2_3 val_a=0.858 test_a=0.838
        * 5-5 256-11-6  (32 32) svs_svd_2_4 val_a=0.481  test_a=
            conc=11 capas lstm es mucho
        * 4-3 256-12-10  (64 64) svs_svd_2_5 val_a=0.48@29  test_a=
        * 5-5 256-7-12  (32 32) svs_svd_2_6 val_a=  test_a=    conc=
        * 4=3 256-2-23  (32 32) att_pool svs_svd_2_7 val_a@3=0.656
            val_a@62=0.698  val_a@63=0.733  val_a@64=0.745
            val_a@276=0.8742 val_a@362=0.882 test_a@276=0.8632

    
    * Conv1d
        * 256-4000-0.1   svs_svd_2_2    val_a=0.57
        * 2048-3000-0.1 svs_svd_2_4 val_a=0.487
        
    * Conv1d + LSTM
        * 768-512-8-2-2-4   svs_svd_2_4 val_a=0.48
        * svs_svd_2_10    val_a=0.48
        * svs_svd_2_3   val_a=0.48
        
* Correr LSTM para ver como funciona
    * Disminuir algo para que quepa en VRAM


optimizar gmm para meter más frames dentro de cada GMM


Depr:
* Usar VoiceActivation para tomar ciertas partes de la cancion completa.
    * Tiene que correr sobre canciones mixeadas () y de-mixeadas (svs_openunmix).
    

Done:
* 
* Ver si el limite de frames de entrenamient de GMM es por clase o en total. (es por clase y fue deprecado)
* Generar evaluación stándar de distintos modelos
    * MFCC tiene que generar la misma cantidad de samples que la version 1d
    
* Debugeear Wavenet + Transformer y LSTM con dummy

* Sacar los silencios de las canciones.

*  checkear que fundione el sistema para que la misma cancion no quede en sets distintos

* hacer division de sets de canciones previo para enforzar:
    1. No quede la misma cancion en multiples sets.
    2. Fijar la eleccion de sets entre multiples experimentos y modelos.
    
* subsampling para balancear las clases
    1. elegir los N-cantantes con mas canciones
    2. Sub-samplear los n-cantantes a la cantidad del que tenga menos
    
   * Evaluar si es necesario parametrizar las redes de manera independiente

