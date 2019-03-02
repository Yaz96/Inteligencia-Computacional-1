""" 
        Este codigo fue realizado por Clemente Miguel Yáñez Contreras Estudiante de ingenieria en 
        Sistemas digitales y Robotica del Tecnologico de Monterrey campus Monterrey con matricula A00817427
                                                                        Monterrey, Nuevo Leon 1 marzo, 2019


  Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey, me comprometo a que mi actuacion en este examen este regida por la honestidad academica.
   
   Este codigo implementa un algoritmo genetico cuyo objetivo es determinar los pesos y biases de cada neurona o nodo en una red neuronal tambien implementada aqui.
   El algoritmo genetico tiene la opcion de hacer el entrenamiento de la red neuronal con 2 diferentes operadores de cruza y dos diferentes operadores de mutacion.

   Se utilizan las librerias Numpy para la creacion y manipulacion de matrices y operaciones con estas, ademas tambien utilice la libreria random para generar numeros
   enteros y flotantes de forma pseudo aleatoria.

    El programa corta al 66 % el data set para establecerlo como training set y el resto 34 % se queda como un set no visto por la red neuronal y por lo tanto para 
    probar lo aprendido.    

    Para correr el codigo lo unico necesario es situarse en la carpeta del proyecto y correrlo de la siguiente forma: python3 Proyecto1.py . Al correr el proyecto se preguntara si
    se quiere modificar los valores que ya estan predefinidos, una lista se despliega antes con los indices que corresponden a cada variable que puede ser modificada para el entrenamiento
    y la red neuronal.

    En la carpeta del proyecto se incluyen varios archivos con diferentes datasets cabe destacar que cada dataset consta de dos archivos, uno de ellos son los datos de entrada y el segundo
    son las etiquetas de salida con los que se corroboraran los resultados. Si se necesita agregar un dataset que no este en la carpeta, basta con tomar las entradas y las salidas y ponerlas
    en archivos separados, a la hora de comenzar la aplicacion de python tan solo basta presionar la tecla 7 y escribir los nombres del archivo (sin la terminacion .txt) 

    El error que se tomo como un "fitess" se obtuvo realizando un promedio del error cuadratico.

    el objeto llamado individuo tiene 3 atributos fitness, PeBioculta, PebiSalida; el atributo PeBiOculta es un arreglo de tamaño 3 veces el numero de entradas que corresponden a los pesos
    de cada nodo de entrada a cada nodo de la capa oculta y los ultimos tres valores son los bias de cada nodo de la capa oculta. De forma parecida el arreglo PeBiSalida es tambien un arreglo
    con un tamaño de 3 veces el numero de entradas mas 3.

"""

import numpy as np
from random import randint
import random

# variables globales
capaAux=[]   #arreglo auxiliar para crear la capa de entrada

archivoEntradas = "acute-nephritis.txt"
archivoSalidas = "acute-nephritis2.txt"
Xfile = open(archivoEntradas,'r')
Yfile = open(archivoSalidas,'r')

arrayY = [] #arreglo donde almacenaremos el string de etiquetas otorgado por Yfile

FunctActflag = "Sigmoidal" # bandera para determinar cual es la funcion de activacion que se utilizará
FilePointer = 0 # variable enterea utilizada como apuntador del archivo que estara abierto y hacer la partición del train set y el test set
OperadorCruza = "ArithCross"
NumeroIndividuos = 5 # numero de individuos en nuestra poblacion
GenLimite = 100 # limite de generaciones establecido, despues de este limite el algoritmo genetico se detiene y obtiene el error minimo
ErrorGoal = 0.01 # variable utilizada para establecer el limite del error minimo a conseguir
NumEntradas = 0 # variable para almacenar el numero de nodos en la capa de entrada
NumFilasY = 0 # variable que almacena el numero total de filas en un archivo de entrada 
Poblacion = [] # lista que contendra los individuos, esta es una lista de elementos de la clase IndivCapas
NextGen = [] # lista que almacena los individuos de la siguiente generacion, es una lista de los "offsprings"
hijos = [] # lista utilizada para almacenar los hijos mientras mutan o se realizan operaciones de cruza
OperadorMut = "Poly Mutation" 
ProbCruza = 1.0 
ProbMutacion =0.1
IndiceGanador = 0 #indice del individuo correspondiente a tener el minimo error de la generacion final dentro del arreglo Poblacion[]




def detEntradas(): # Esta funcion determina el numero de entradas y el numero de filas que tiene el archivo de entrada
    global NumFilasY
    line = Xfile.readline()
    DataAdq(line)
    NumEntradas = len(capaAux)
    for _ in range(0,len(capaAux)):
            capaAux.pop()
    Xfile.seek(0)
    
    for line in Yfile:
        arrayY.append(float(line[0]))
    NumFilasY =len(arrayY)

    return NumEntradas

    


def DataAdq(line):    # esta funcion toma un renglon de uno de los archivos como entrada y utiliza la variable capaAux para guardar cada valor de este archivo
                    # obteniendo asi un arreglo que luego va a ser renombrado como la capa de entrada de la red neuronal
 
    wraux=''
    for i in range(0, len(line)):
        if (line[i] == '\t' or line[i]==' ' ) and wraux!='':
            capaAux.append(float(wraux))
            wraux =''
        elif line[i] != '\t' or line[i]!=' ' :
            wraux = wraux+line[i]
    capaAux.append(float(wraux))
    #print(capaAux)


def RedNeuronal(Individuo): # esta funcion es la red neuronal que se utiliza para el entrenamiento
    global NumFilasY             ###########
    global FilePointer
    ErrorAcum = 0.0
    WeiMat= np.zeros(( NumEntradas,3))
    WeiMatSalida = np.zeros((3,1))
    Entradas = np.zeros((1,NumEntradas))        # se inicializan las variables que representaran los pesos, las entradas, los bias, las Zs y las Ys de la red Neuronal
    byasArrayOculta = np.array([1,1,1])
    byasSalida = 0
    Zoculta = np.zeros(3)
    Zsalida = 0
    Yoculta = np.zeros(3)
    Ysalida = 0
    listaux = []                #############

                                #Creacion de la matriz de pesos Capa Oculta
    for j in range(0,NumEntradas): #iteracion por nodo 
        listaux = Individuo.getPesoculta(j)
        for k in range(0,3): # iteracioon por peso de un nodo 
            WeiMat[j][k] = listaux[k]


                                    #Creacion del arreglo de Bias capa Oculta
    listaux = Individuo.getBiasOculta() 
    for j in range(0,3):
        byasArrayOculta[j] = listaux[j]

  
    
         #Creacion de la matriz de pesos capa Salida
    listaux = Individuo.getPeBiSalida()
    for k in range(0,3): # iteracioon por nodo 
        WeiMatSalida[k][0] = listaux[k]



    byasSalida = listaux[3]  #Creacion de la constante Bias de salida

 
    
    ContadorLinea = 0           # contador de la linea en la que vamos, necesario para usarlo como indice y comparar la salida de la red neuronal con la salida real
    Limite = int(NumFilasY*.66-1) # Esta variable determina el 66% del data set que corresponde al entrenamiento

    while(ContadorLinea<=Limite):  # este loop inicializa la propagacion de la red neuronal
    #for line in Xfile:
        line = Xfile.readline()
        DataAdq(line)

                    #Creacion arreglo de Entradas
        for x in range(NumEntradas):
            Entradas[0,x]=capaAux[0]
            capaAux.pop(0)
    

        Zoculta = Entradas.dot(WeiMat)+ byasArrayOculta 
      

        if FunctActflag == "Sigmoidal":
            for x in range(3):
                Yoculta[x] = 1.0/(1.0+np.exp(-1.0*Zoculta[0][x]))
                
        elif FunctActflag == "Tanh" :
            for x in range(3):
                Yoculta[x] = 2.0/(1.0+np.exp(-1.0*Zoculta[0][x])) -1.0

        Zsalida = Yoculta.dot(WeiMatSalida)+ byasSalida

        Ysalida = 1.0/(1.0+np.exp(-1.0*Zsalida))

        ErrorAcum= ErrorAcum + (Ysalida - arrayY[ContadorLinea])**2
        ContadorLinea = ContadorLinea+1
        if ContadorLinea == Limite :  
            if (FilePointer == 0):    
                FilePointer=Xfile.tell()
    
    
    Xfile.seek(0)
    ErrorAcum = ErrorAcum/ContadorLinea
    #Individuo.setFitness(ErrorAcum)

    return ErrorAcum

def RedNeuronalTesteo(Individuo): # esta red neuronal se utiliza para el testeo de los pesos y bias adquiridos
    global NumFilasY
    global FilePointer
    ErrorAcum = 0.0
    WeiMat= np.zeros(( NumEntradas,3))
    WeiMatSalida = np.zeros((3,1))
    Entradas = np.zeros((1,NumEntradas))
    byasArrayOculta = np.array([1,1,1])
    byasSalida = 0
    Zoculta = np.zeros(3)
    Zsalida = 0
    Yoculta = np.zeros(3)
    Ysalida = 0
    listaux = []
                                #Creacion de la matriz de pesos Capa Oculta
    for j in range(0,NumEntradas): #iteracion por nodo 
        listaux = Individuo.getPesoculta(j)
        for k in range(0,3): # iteracioon por peso de un nodo 
            WeiMat[j][k] = listaux[k]


                                    #Creacion del arreglo de Bias capa Oculta
    listaux = Individuo.getBiasOculta() 
    for j in range(0,3):
        byasArrayOculta[j] = listaux[j]

   
    
         #Creacion de la matriz de pesos capa Salida
    listaux = Individuo.getPeBiSalida()
    for k in range(0,3): # iteracioon por nodo 
        WeiMatSalida[k][0] = listaux[k]



    byasSalida = listaux[3]  #Creacion de la constante Bias de salida
    
    
    ContadorLinea = int(NumFilasY*.66)
    
    Xfile.seek(FilePointer)
    while(ContadorLinea<=NumFilasY-1):

        line = Xfile.readline()
        DataAdq(line)
        #Creacion arreglo de Entradas
        
        for x in range(NumEntradas):
            Entradas[0,x]=capaAux[0]
            capaAux.pop(0)
        

        Zoculta = Entradas.dot(WeiMat)+ byasArrayOculta
      

        if FunctActflag == "Sigmoidal":
            for x in range(3):
                Yoculta[x] = 1.0/(1.0+np.exp(-1.0*Zoculta[0][x]))
                
        elif FunctActflag == "Tanh" :
            for x in range(3):
                Yoculta[x] = 2.0/(1.0+np.exp(-1.0*Zoculta[0][x])) -1.0

        Zsalida = Yoculta.dot(WeiMatSalida)+ byasSalida

        Ysalida = 1.0/(1.0+np.exp(-1.0*Zsalida))

        ErrorAcum= ErrorAcum + (Ysalida - arrayY[ContadorLinea])**2
        ContadorLinea = ContadorLinea+1
    
    ErrorAcum = ErrorAcum/ContadorLinea
    #Individuo.setFitness(ErrorAcum)

    return ErrorAcum



class IndivCapas():             # esta es la clase utilizada para cada individuo
    PesosBiasOculta = []
    PesosBiasSalida = []
    fitness = 0
    
    def __init__(self,Entradas, fit = 1000 ):
        self.PesosBiasOculta  = np.zeros(NumEntradas*3+3) 
        self.PesosBiasSalida = np.zeros(4)
        self.fitness = fit

    def getPeBiOculta(self): # regresa el arreglo completo
        return self.PesosBiasOculta

    def getFitness(self): # regresa el fitness
        return self.fitness
    
    def getPeBiSalida(self): # regresa el arreglo completo 
        return  self.PesosBiasSalida
    def setPeBiOcultaCom(self, pebioc): # modifica el arreglo completo
        self.PesosBiasOculta = pebioc
    def setPeBiOcultaIndiv(self,NumElemento, numero): # modifica el atributo en un determinado elemento
        self.PesosBiasOculta[NumElemento] = numero
    
    def setPeBiOculta(self, arreglo,NumNodo):  # modifica el arreglo por numero de nodos y recibe un arreglo de 3 valores
        for x in range(0,3):
            self.PesosBiasOculta[x+3*NumNodo] = arreglo[x]
        
    def setPeBiSalida(self, arreglo):  # modifica el atributo de la capa de salida completa
        for x in range(0,4):
            self.PesosBiasSalida[x] = arreglo[x]
    def setPeBiSalidaIndiv(self, NumElemento, numero): #  # modifica el atributo de la capa de salida de forma individual cada elemento
        self.PesosBiasSalida[NumElemento] = numero
    
    def setFitness(self, fit): # modifica el fitness de un elemento
        self.fitness = fit


    def getPesoculta(self, nodoNum): # regresa los pesos de un nodo individual
        lista = np.zeros(3) 
        for x in range(3):
            lista[x]=self.PesosBiasOculta[nodoNum*3+x]
        return lista
    def getBiasOculta(self): # regresa los pesos de un nodo individual
        lista = np.zeros(3) 
        lista[0]=self.PesosBiasOculta[-3]
        lista[1]=self.PesosBiasOculta[-2]
        lista[2]=self.PesosBiasOculta[-1]
        return lista

    def getPeBiOcultaUnoxUno(self, i): #regresa un peso determinado
        return self.PesosBiasOculta[i]

    def getPeBiSalidaUnoxUno(self, i): #regresa un peso determinado
        return self.PesosBiasSalida[i]
    def getPeBiOcultaComp(self): #regresa la capa completa
        return self.PesosBiasOculta
    



def InitPoblacion():  # esta funcion inicializa la poblacion y el arreglo Next con numero random
    lista=[0.0,0.0,0.0,0.0]
    for _ in range(NumeroIndividuos):
        Poblacion.append(IndivCapas(Entradas=NumEntradas))
    
    for _ in range(NumeroIndividuos):
        NextGen.append(IndivCapas(Entradas=NumEntradas))
    for _ in range(0,2):
        hijos.append(IndivCapas(Entradas=NumEntradas))
    

    for x in range(NumeroIndividuos):      #----------------------- llenado de la capa oculta 
        for y in range(0,NumEntradas + 1):
            lista[0]=random.randint(-8,8)
            lista[1]=random.randint(-8,8)
            lista[2]=random.randint(-8,8)
            Poblacion[x].setPeBiOculta(lista,y)

    for x in range(NumeroIndividuos):      #----------------------- llenado de la capa salida 
        
        lista[0]=random.randint(-20,20)
        lista[1]=random.randint(-20,20)
        lista[2]=random.randint(-20,20)
        lista[3]=random.randint(-20,20)
        Poblacion[x].setPeBiSalida(lista)

def DetErrorMin(): # esta funcion determina el error minimo
    ErrorM = [1000.0,0]

    for i in range(NumeroIndividuos):
        if(Poblacion[i].getFitness()<ErrorM[0]):
            ErrorM = [Poblacion[i].getFitness(),i]
    return ErrorM


def Tournament():  # esta funcion hace torneo entre 4 padres y selecciona los dos mejores
    padres = [0,0]
    while padres[0] == padres[1]:
        randAux1 = random.randint(0,NumeroIndividuos-1) 
        randAux2 = random.randint(0,NumeroIndividuos-1)
        while randAux1 == randAux2:
            randAux2 = random.randint(0,NumeroIndividuos-1)
            

        if Poblacion[randAux1].getFitness() < Poblacion[randAux2].getFitness():
            padres[0] = randAux1

        elif Poblacion[randAux1].getFitness() >= Poblacion[randAux2].getFitness():
            padres[0] = randAux2
        else:
            print("Error in Tournament padre1")
        
        randAux1 = random.randint(0,NumeroIndividuos-1)
        randAux2 = random.randint(0,NumeroIndividuos-1)
        while randAux1 == randAux2:
            randAux2 = random.randint(0,NumeroIndividuos-1)
        
        if Poblacion[randAux1].getFitness() < Poblacion[randAux2].getFitness():
            padres[1] = randAux1

        elif Poblacion[randAux1].getFitness() >= Poblacion[randAux2].getFitness():
            padres[1] = randAux2
        else:
            print("Error in Tournament padre2")

    
    if Poblacion[padres[0]].getFitness()> Poblacion[padres[1]].getFitness():
        numaux=padres[0]
        padres[0] = padres[1]
        padres[1] = numaux



    return padres
    




def AlgoritmoGenetico(): # esta es la funcion del algoritmo genetico 

    global IndiceGanador  
    global GenLimite
    global ErrorGoal
    hijo = IndivCapas(Entradas=NumEntradas) # este es el hijo que se conserva despues de la cruza y que potencialmente sera mutado
    generaciones = 0  
    padres =[]
    deltaMax = 10 
    sigma = 2

    #xAxis = []
    #yAxis = []
    

    for x in range(NumeroIndividuos):        #evaluate the fitness of each individual of the first gen
        Poblacion[x].setFitness( RedNeuronal(Poblacion[x]))

    
   
    
    StopCriteria = True
    ErrorMin =  [1000.0,0] 
    ErrorAnter = [1000.0,0]
    ContadorRep = 0
    
    while (StopCriteria) :
        
        for y in range(NumeroIndividuos) :     
            if  np.random.uniform(0,1) < ProbCruza:        
                padres = Tournament()           
                
                auxlistOculta1= np.zeros(NumEntradas*3+3)
                auxlistOculta2= np.zeros(NumEntradas*3+3)
                if OperadorCruza == "ArithCross":     # aqui comienza la cruza aritmetica
                    # alpha = 0.5 alpha*p1 + (1-alpha)*p2 
                    for j in range(0,NumEntradas*3+3):  # aqui se realiza la cruza de los dos padres de su capa oculta
                        alpha = np.random.uniform(0,1)
                        auxlistOculta1[j] = Poblacion[padres[0]].getPeBiOcultaUnoxUno(j)*alpha + (1-alpha)*Poblacion[padres[1]].getPeBiOcultaUnoxUno(j)
                        auxlistOculta2[j] = Poblacion[padres[1]].getPeBiOcultaUnoxUno(j)*alpha + (1-alpha)*Poblacion[padres[0]].getPeBiOcultaUnoxUno(j)
                        

                    hijos[0].setPeBiOcultaCom(auxlistOculta1)
                    hijos[1].setPeBiOcultaCom(auxlistOculta2)

                    auxlistaSalida1 = np.zeros(4)
                    auxlistaSalida2 = np.zeros(4)
                    for j in range(4):      # aqui se realiza la cruza de los dos padres de su capa de salida
                        alpha = np.random.uniform(0,1)
                        auxlistaSalida1[j] = Poblacion[padres[0]].getPeBiSalidaUnoxUno(j)*alpha + (1-alpha)*Poblacion[padres[1]].getPeBiSalidaUnoxUno(j)
                        auxlistaSalida2[j] = Poblacion[padres[1]].getPeBiSalidaUnoxUno(j)*alpha + (1-alpha)*Poblacion[padres[0]].getPeBiSalidaUnoxUno(j)

                    hijos[0].setPeBiSalida(auxlistaSalida1)
                    hijos[1].setPeBiSalida(auxlistaSalida2)

                elif OperadorCruza == "SBX": # aqui comienza la cruza SBX 
                    
                    auxlistOculta1= np.zeros(NumEntradas*3+3)
                    auxlistOculta2= np.zeros(NumEntradas*3+3)
                    for j in range(0,NumEntradas*3+3):  # aqui se realiza la cruza de los dos padres de su capa oculta
                        u= np.random.uniform(0,1)
                        if u <= 0.5:
                            Bi= (2*u)**(1/(generaciones+1))
                        else: 
                            Bi = (2*(1-u))**(-1/(generaciones+1))
                        auxlistOculta1[j] = 0.5*(Poblacion[padres[0]].getPeBiOcultaUnoxUno(j)+Poblacion[padres[1]].getPeBiOcultaUnoxUno(j) ) - .5 * Bi *(Poblacion[padres[0]].getPeBiOcultaUnoxUno(j)-Poblacion[padres[1]].getPeBiOcultaUnoxUno(j))
                        auxlistOculta2[j] = 0.5*(Poblacion[padres[0]].getPeBiOcultaUnoxUno(j)+Poblacion[padres[1]].getPeBiOcultaUnoxUno(j) ) - .5 * Bi *(Poblacion[padres[1]].getPeBiOcultaUnoxUno(j)-Poblacion[padres[0]].getPeBiOcultaUnoxUno(j))

                    hijos[0].setPeBiOcultaCom(auxlistOculta1)
                    hijos[1].setPeBiOcultaCom(auxlistOculta2)

                    auxlistaSalida1 = np.zeros(4)
                    auxlistaSalida2 = np.zeros(4)
                    for j in range(4):                  # aqui se realiza la cruza de los dos padres de su capa de salida
                        u= np.random.uniform(0,1)
                        if u <= 0.5:
                            Bi= (2*u)**(1/(generaciones+1))
                        else: 
                            Bi = (2*(1-u))**(-1/(generaciones+1))

                        auxlistaSalida1[j] = 0.5*(Poblacion[padres[0]].getPeBiSalidaUnoxUno(j)+Poblacion[padres[1]].getPeBiSalidaUnoxUno(j) ) - .5 * Bi *(Poblacion[padres[0]].getPeBiSalidaUnoxUno(j)-Poblacion[padres[1]].getPeBiSalidaUnoxUno(j))
                        auxlistaSalida1[j] = 0.5*(Poblacion[padres[0]].getPeBiSalidaUnoxUno(j)+Poblacion[padres[1]].getPeBiSalidaUnoxUno(j) ) - .5 * Bi *(Poblacion[padres[1]].getPeBiSalidaUnoxUno(j)-Poblacion[padres[0]].getPeBiSalidaUnoxUno(j))

                    hijos[0].setPeBiSalida(auxlistaSalida1)
                    hijos[1].setPeBiSalida(auxlistaSalida2)

                    
            
            if OperadorMut == "Poly Mutation":  # Aqui comienza la mutacion polinomial
                for s in range(0,2):
                    for x in range(0,NumEntradas*3+3):
                        if np.random.uniform(0,1) < ProbMutacion:
                            u= np.random.uniform(0,1)
                            if u <= 0.5:
                                Bi= (2*u)**(1/(generaciones+1)) -1
                            else: 
                                Bi = 1-(2*(1-u))**(-1/(generaciones+1))
                            hijos[s].setPeBiOcultaIndiv(x,hijos[s].getPeBiOcultaUnoxUno(x)+ deltaMax*Bi )
                
                for s in range(0,2):
                    for x in range(0,4):
                        if np.random.uniform(0,1) < ProbMutacion:
                            u= np.random.uniform(0,1)
                            if u <= 0.5:
                                Bi= (2*u)**(1/(generaciones+1))-1
                            else: 
                                Bi = 1-(2*(1-u))**(-1/(generaciones+1))
                            hijos[s].setPeBiSalidaIndiv(x,hijos[s].getPeBiSalidaUnoxUno(x)+ deltaMax*Bi )

                
            elif OperadorMut == "Normal Mutation":  # Aqui comienza la mutacion Normal
                for s in range(0,2):
                    for x in range(0,NumEntradas*3+3):
                        if np.random.uniform(0,1) < ProbMutacion:    
                            hijos[s].setPeBiOcultaIndiv(x,hijos[s].getPeBiOcultaUnoxUno(x)+ sigma* np.random.randn() )
                
                for s in range(0,2):
                    for x in range(0,4):
                        if np.random.uniform(0,1) < ProbMutacion:                            
                            hijos[s].setPeBiSalidaIndiv(x,hijos[s].getPeBiSalidaUnoxUno(x)+  sigma* np.random.randn())


            
            hijos[0].setFitness(RedNeuronal(hijos[0]))
            hijos[1].setFitness(RedNeuronal(hijos[1]))
           
                
            if hijos[0].getFitness() < hijos[1].getFitness(): # este if compara el hijo 1 con el hijo 2 y escribe el de mejor fitness en la variable hijo
                hijo = hijos[0]
            else: 
                hijo = hijos[1]
            


            if(Poblacion[padres[0]].getFitness() < Poblacion[padres[1]].getFitness()): # en estos if se compara al mejor padre con el hijo y nos quedamos con el mejor escribiendolo en el arreglo de siguiente generacion
                if(Poblacion[padres[0]].getFitness()> hijo.getFitness()):
                    NextGen[y].setPeBiOcultaCom(hijo.getPeBiOcultaComp())
                    NextGen[y].setPeBiSalida(hijo.getPeBiSalida()) 
                    NextGen[y].setFitness(hijo.getFitness())
                    
                else:
                    NextGen[y].setPeBiOcultaCom(Poblacion[padres[0]].getPeBiOcultaComp())
                    NextGen[y].setPeBiSalida(Poblacion[padres[0]].getPeBiSalida()) 
                    NextGen[y].setFitness(Poblacion[padres[0]].getFitness())
                
            else:
                if(Poblacion[padres[1]].getFitness()> hijo.getFitness()):
                    NextGen[y].setPeBiOcultaCom(hijo.getPeBiOcultaComp()) 
                    NextGen[y].setPeBiSalida(hijo.getPeBiSalida()) 
                    NextGen[y].setFitness(hijo.getFitness())
                else:
                    NextGen[y].setPeBiOcultaCom(Poblacion[padres[1]].getPeBiOcultaComp()) 
                    NextGen[y].setPeBiSalida(Poblacion[padres[1]].getPeBiSalida()) 
                    NextGen[y].setFitness(Poblacion[padres[1]].getFitness())

           
        #----------------- ------------------------------------------------------------------------------------------------------------------------------

       
        
        
        for x in range(NumeroIndividuos):  # aqui se pasa la generacion antigua con la actual
            Poblacion[x].setPeBiOcultaCom( NextGen[x].getPeBiOcultaComp() )
            Poblacion[x].setPeBiSalida( NextGen[x].getPeBiSalida())
            Poblacion[x].setFitness( NextGen[x].getFitness() )


        ErrorMin = DetErrorMin() # esta  linea regresa el error minimo en un arreglo 
        
        
        generaciones = generaciones +1
        
       


        if (generaciones>GenLimite): # en estos if se compara cada variable correspondiente con sus criterios de paro
            StopCriteria = False
            print("Error Minimo: "+str(ErrorMin[0])+ "  Corresponde al individuo "+ str(ErrorMin[1]) + "  Salida por generaciones")
            print(generaciones, " generaciones")
        if ErrorMin[0]<ErrorGoal:
            StopCriteria = False
            print("Error Minimo: "+str(ErrorMin[0])+ "  Corresponde al individuo "+ str(ErrorMin[1]) + "  Salida por Error Minimo Cumplido")
            print(generaciones, " generaciones")

        if ContadorRep >20 :
            StopCriteria = False
            print("Error Minimo: "+str(ErrorMin[0])+ "  Corresponde al individuo "+ str(ErrorMin[1]) + "  Salida por Minimo Estancado")
            print(generaciones, " generaciones")

        if ErrorMin[0] == ErrorAnter[0]:
            ContadorRep = ContadorRep +1
        else:
            ContadorRep = 0

        ErrorAnter = ErrorMin # esta asignacion es hecha para tener en cuenta cuantas veces se ha repetido el mismo valor del error y determinar un estancamiento

    
    IndiceGanador = ErrorMin[1] # al final se asigna al inidice ganador el error minimo de la ultima generacion
    print(Poblacion[ErrorMin[1]].getFitness(), "Fitness Final") # y se despliega este valor de fitness
    





def FuncionTesteo(): # esta funcion testea los pesos y bias entrenados anteriormente
    global IndiceGanador
    ErrorTesteo = RedNeuronalTesteo(Poblacion[IndiceGanador]) 

    print(ErrorTesteo, " Error en el test ", Poblacion[IndiceGanador].getFitness(), " Error en el entrenamiento") # se imprime una comparacion entre el error en el test y el del entrenamiento

def Menu():                      # esta funcion es el menu que aparece al comienzo y que determina los datos con los que se entrenara la red neuronal
    global OperadorMut
    global OperadorCruza
    global FunctActflag
    global NumeroIndividuos
    global GenLimite
    global ErrorGoal
    global archivoEntradas
    global archivoSalidas
    global ProbCruza
    global ProbMutacion
    global Xfile,Yfile

    print(" ")
    print("Esta es la configuracion estandar: ")
    print(" ")
    print("1. OPERADOR MUTACION => "+ OperadorMut)
    print("2. OPERADOR CRUZA => " + OperadorCruza)
    print("3. FUNCION DE ACTIVACION DE CAPA OCULTA => " + FunctActflag )
    print("4. POBLACION => " + str(NumeroIndividuos) )
    print("5. NUMERO DE GENERACIONES LIM =>  " + str(GenLimite) )
    print("6. ERROR MIN (Target) => " + str(ErrorGoal) )
    print("7. ARCHIVO DE ENTRADAS => " + archivoEntradas )
    print("   ARCHIVO DE SALIDAS(etiquetas): " + archivoSalidas)
    print("8. PROBABILIDAD DE MUTACION =>  " + str(ProbMutacion) )
    print(" ")
    print(" ")
    indice = int(input("Que numero deseas cambiar?  0 para continuar sin cambios "))
    if indice != 0:

        while(indice != 0):
            
            if indice == 1:
                OperadorMut=input("Operador Mutacion: Poly Mutation || Normal Mutation   ")
                while not(OperadorMut == "Poly Mutation" or OperadorMut == "Normal Mutation" ):
                    OperadorMut=input("Operador Mutacion: Poly Mutation || Normal Mutation   ")
            elif indice == 2:
                OperadorCruza=input("Operador Cruza: SBX || ArithCross   ")
                while not(OperadorCruza == "SBX" or OperadorCruza == "ArithCross" ):
                    OperadorCruza=input("Operador Cruza: SBX || ArithCross   ")
            elif indice == 3:
                FunctActflag = input("Funcion de Activacion: Tanh || Sigmoidal   ")
                while not(FunctActflag == "Tanh" or FunctActflag == "Sigmoidal"  ):
                    FunctActflag = input("Funcion de Activacion: Tanh || Sigmoidal   ")
            elif indice == 4:
                NumeroIndividuos = int(input("Numero de Individuos por Generacion (Tamaño de la poblacion[minimo 3]) :  "))
            elif indice == 5:
                GenLimite = int(input("Numero Generaciones Limite:  "))
            elif indice == 6:
                ErrorGoal = float(input("Error minimo a alcanzar:  "))
            elif indice == 7:
                archivoEntradas = input("Nombre del archivo con las entradas sin el .txt:  ") + ".txt"
                Xfile.close()
                Xfile = open(archivoEntradas,'r')
                archivoSalidas = input("Nombre del archivo con las etiquetas sin el .txt:  ") + ".txt"
                Yfile.close()
                Yfile = open(archivoSalidas,'r')

            elif indice == 8:
                ProbCruza = int(input("Probabilidad de Mutacion: "))
            
            indice = int(input("Que numero deseas cambiar?  0 para continuar sin cambios "))
    
    print("Esta es la configuracion actual: ")
    print(" ")
    print("1. OPERADOR MUTACION => "+ OperadorMut)
    print("2. OPERADOR CRUZA => " + OperadorCruza)
    print("3. FUNCION DE ACTIVACION DE CAPA OCULTA => " + FunctActflag )
    print("4. POBLACION => " + str(NumeroIndividuos) )
    print("5. NUMERO DE GENERACIONES LIM =>  " + str(GenLimite) )
    print("6. ERROR MIN (Target) => " + str(ErrorGoal) )
    print("7. ARCHIVO DE ENTRADAS => " + archivoEntradas )
    print("   ARCHIVO DE SALIDAS(etiquetas): " + archivoSalidas)
    print("8. PROBABILIDAD DE MUTACION =>  " + str(ProbMutacion) )
    print(" ")
    print(" ")
    print("PROCESANDO...")
    print(" ")
    print(" ")









    

# aqui comienza el main ()
    


Menu() # primero se manda a llamar a la funcion del menu
NumEntradas = detEntradas() # despues se determina cuantas entradas tiene el archivo y cuantas filas
InitPoblacion() # tercero se inicializa la primera generacion con valores aleatorios
AlgoritmoGenetico() # mas tarde se llama al entrenamiento desde el algoritmo genetico 
FuncionTesteo() # una vez obtenido los valore de pesos y bias se hace la funcion de testeo 
Xfile.close() # se cierran finalmente los dos archivos que habiamos abierto antes
Yfile.close()





