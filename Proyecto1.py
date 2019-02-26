import numpy as np
from random import randint
import random

capaAux=[]
Xfile = open("X1.txt",'r')
Yfile = open("Y1.txt",'r')
arrayY = []
FunctActflag = "Sigmoidal"

OperadorCruza = "ArithCross"
NumeroIndividuos = 50
NumEntradas = 0
Poblacion = []
NextGen = []
hijos = []
OperadorMut = "Bit Flip"
ProbCruza = 1.0
ProbMutacion =0.1


def detEntradas():
    line = Xfile.readline()
    DataAdq(line)
    NumEntradas = len(capaAux)
    for _ in range(0,len(capaAux)):
            capaAux.pop()
    Xfile.seek(0)
    
    for line in Yfile:
        arrayY.append(float(line[0]))
    return NumEntradas

    


def DataAdq(line):    
 
    wraux=''
    for i in range(0, len(line)):
        if (line[i] == '\t' or line[i]==' ' ) and wraux!='':
            capaAux.append(float(wraux))
            wraux =''
        elif line[i] != '\t' or line[i]!=' ' :
            wraux = wraux+line[i]
    capaAux.append(float(wraux))
    #print(capaAux)


def RedNeuronal(Individuo):
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
    #print("Weimat"+ '\n', WeiMat)

                                    #Creacion del arreglo de Bias capa Oculta
    listaux = Individuo.getBiasOculta() 
    for j in range(0,3):
        byasArrayOculta[j] = listaux[j]

    #print("byasOculata"+ '\n', byasArrayOculta)
    
         #Creacion de la matriz de pesos capa Salida
    listaux = Individuo.getPeBiSalida()
    for k in range(0,3): # iteracioon por nodo 
        WeiMatSalida[k][0] = listaux[k]

    #print("Pesos Salida"+ '\n', WeiMatSalida)

    byasSalida = listaux[3]  #Creacion de la constante Bias de salida
    #print("Byas Salida"+ '\n', byasSalida)
    

 

    ContadorLinea = 0
    for line in Xfile:
        DataAdq(line)
        #Creacion arreglo de Entradas
        
        for x in range(NumEntradas):
            Entradas[0,x]=capaAux[0]
            capaAux.pop(0)
        #print("Entradas"+'\n', Entradas)

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
    Xfile.seek(0)
    ErrorAcum = ErrorAcum/ContadorLinea
    #Individuo.setFitness(ErrorAcum)
    return ErrorAcum



class IndivCapas():
    PesosBiasOculta = []
    PesosBiasSalida = []
    fitness = 0
    
    def __init__(self,Entradas, fit = 1000 ):
        self.PesosBiasOculta  = np.zeros(NumEntradas*3+3) 
        self.PesosBiasSalida = np.zeros(4)
        self.fitness = fit

    def getPeBiOculta(self):
        return self.PesosBiasOculta

    def getFitness(self):
        return self.fitness
    
    def getPeBiSalida(self):
        return  self.PesosBiasSalida
    def setPeBiOcultaCom(self, pebioc):
        self.PesosBiasOculta = pebioc
    
    def setPeBiOculta(self, arreglo,NumNodo):
        for x in range(0,3):
            self.PesosBiasOculta[x+3*NumNodo] = arreglo[x]
        
    def setPeBiSalida(self, arreglo):
        for x in range(0,4):
            self.PesosBiasSalida[x] = arreglo[x]
    
    def setFitness(self, fit):
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

    def getPeBiOcultaUnoxUno(self, i):
        return self.PesosBiasOculta[i]

    def getPeBiSalidaUnoxUno(self, i):
        return self.PesosBiasSalida[i]
    def getPeBiOcultaComp(self):
        return self.PesosBiasOculta
    



def InitPoblacion():
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

def DetErrorMin():
    ErrorM = [1000.0,0]

    for i in range(NumeroIndividuos):
        if(Poblacion[i].getFitness()<ErrorM[0]):
            ErrorM = [Poblacion[i].getFitness(),i]
    return ErrorM


def Tournament():
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
    




def AlgoritmoGenetico():
    alpha = 0.5
    hijo1 = IndivCapas(Entradas=NumEntradas)

    hijo = IndivCapas(Entradas=NumEntradas)
    generaciones = 0 
    padres =[]

    #xAxis = []
    #yAxis = []
    

    for x in range(NumeroIndividuos):  #evaluate the fitness of each individual of the first gen
        Poblacion[x].setFitness( RedNeuronal(Poblacion[x]))

    #for x in range(NumeroIndividuos):
    #    print(Poblacion[x].getFitness()   )
    
    StopCriteria = True
    ErrorMin =  [1000.0,0]
    ErrorAnter = [1000.0,0]
    ContadorRep = 0
    while (StopCriteria) :
        
        for y in range(NumeroIndividuos):
            if  np.random.uniform(0,1) < ProbCruza:
                padres = Tournament()
                
                auxlistOculta1= np.zeros(NumEntradas*3+3)
                auxlistOculta2= np.zeros(NumEntradas*3+3)
                if OperadorCruza == "ArithCross":
                    # alpha = 0.5 alpha*p1 + (1-alpha)*p2 
                    for j in range(0,NumEntradas*3+3): 
                        alpha = np.random.uniform(0,1)
                        auxlistOculta1[j] = Poblacion[padres[0]].getPeBiOcultaUnoxUno(j)*alpha + (1-alpha)*Poblacion[padres[1]].getPeBiOcultaUnoxUno(j)
                        auxlistOculta2[j] = Poblacion[padres[1]].getPeBiOcultaUnoxUno(j)*alpha + (1-alpha)*Poblacion[padres[0]].getPeBiOcultaUnoxUno(j)
                        

                    hijos[0].setPeBiOcultaCom(auxlistOculta1)
                    hijos[1].setPeBiOcultaCom(auxlistOculta2)

                    auxlistaSalida1 = np.zeros(4)
                    auxlistaSalida2 = np.zeros(4)
                    for j in range(4):
                        alpha = np.random.uniform(0,1)
                        auxlistaSalida1[j] = Poblacion[padres[0]].getPeBiSalidaUnoxUno(j)*alpha + (1-alpha)*Poblacion[padres[1]].getPeBiSalidaUnoxUno(j)
                        auxlistaSalida2[j] = Poblacion[padres[1]].getPeBiSalidaUnoxUno(j)*alpha + (1-alpha)*Poblacion[padres[0]].getPeBiSalidaUnoxUno(j)

                    hijos[0].setPeBiSalida(auxlistaSalida1)
                    hijos[1].setPeBiSalida(auxlistaSalida2)

                elif OperadorCruza == "SBX": # no usar esta 
                    
                    auxlistOculta1= np.zeros(NumEntradas*3+3)
                    auxlistOculta2= np.zeros(NumEntradas*3+3)
                    for j in range(0,NumEntradas*3+3): 
                        u= np.random.uniform(0,1)
                        if u <= 0.5:
                            Bi= (2*u)**(1/generaciones+1)
                        else: 
                            Bi = (2*(1-u))**(-1/generaciones+1)
                        auxlistOculta1[j] = 0.5*(Poblacion[padres[0]].getPeBiOcultaUnoxUno(j)+Poblacion[padres[1]].getPeBiOcultaUnoxUno(j) ) - .5 * Bi *(Poblacion[padres[0]].getPeBiOcultaUnoxUno(j)-Poblacion[padres[1]].getPeBiOcultaUnoxUno(j))
                        auxlistOculta2[j] = 0.5*(Poblacion[padres[0]].getPeBiOcultaUnoxUno(j)+Poblacion[padres[1]].getPeBiOcultaUnoxUno(j) ) - .5 * Bi *(Poblacion[padres[1]].getPeBiOcultaUnoxUno(j)-Poblacion[padres[0]].getPeBiOcultaUnoxUno(j))

                    hijos[0].setPeBiOcultaCom(auxlistOculta1)
                    hijos[1].setPeBiOcultaCom(auxlistOculta2)

                    auxlistaSalida1 = np.zeros(4)
                    auxlistaSalida2 = np.zeros(4)
                    for j in range(4):
                        u= np.random.uniform(0,1)
                        if u <= 0.5:
                            Bi= (2*u)**(1/generaciones+1)
                        else: 
                            Bi = (2*(1-u))**(-1/generaciones+1)

                        auxlistaSalida1[j] = 0.5*(Poblacion[padres[0]].getPeBiSalidaUnoxUno(j)+Poblacion[padres[1]].getPeBiSalidaUnoxUno(j) ) - .5 * Bi *(Poblacion[padres[0]].getPeBiSalidaUnoxUno(j)-Poblacion[padres[1]].getPeBiSalidaUnoxUno(j))
                        auxlistaSalida1[j] = 0.5*(Poblacion[padres[0]].getPeBiSalidaUnoxUno(j)+Poblacion[padres[1]].getPeBiSalidaUnoxUno(j) ) - .5 * Bi *(Poblacion[padres[1]].getPeBiSalidaUnoxUno(j)-Poblacion[padres[0]].getPeBiSalidaUnoxUno(j))

                    hijos[0].setPeBiSalida(auxlistaSalida1)
                    hijos[1].setPeBiSalida(auxlistaSalida2)

                    
            if np.random.uniform(0,1) < ProbMutacion:
                if OperadorMut == "Bit Flip":
                    auxlistOculta= np.zeros(NumEntradas*3+3)
                    Mask=random.randint(-1000,1000 )
                    for j in range(0,NumEntradas*3+3): 
                        Mask=random.randint(-1000,1000)
                        auxlistOculta[j] = int (np.round(NextGen[y].getPeBiOcultaUnoxUno(j) ))  ^ Mask
                
                    NextGen[y].setPeBiOcultaCom(auxlistOculta)
                    Mask=random.randint(-1000,1000 )
                    auxlistaSalida = np.zeros(4)
                    for j in range(4):
                        Mask=random.randint(-1000,1000 )
                        auxlistaSalida[j] =  int (np.round(NextGen[y].getPeBiSalidaUnoxUno(j) ))  ^ Mask
                    NextGen[y].setPeBiSalida(auxlistaSalida)
                elif OperadorMut == "Real Value Encoding":
                    print("Real Value Encoding")


            
            hijos[0].setFitness(RedNeuronal(hijos[0]))
            hijos[1].setFitness(RedNeuronal(hijos[1]))
                
            if hijos[0].getFitness() < hijos[1].getFitness():
                hijo = hijos[0]
            else: 
                hijo = hijos[1]

            if(Poblacion[padres[0]].getFitness() < Poblacion[padres[1]].getFitness()):
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
                    
                
                #elif OperadorMut == "PolyMutation":
        
        for x in range(NumeroIndividuos):
            Poblacion[x] = NextGen[x]

                    







        
        generaciones = generaciones +1
        ErrorMin = DetErrorMin()
        
        #print(ErrorMin)
        #if (generaciones%10==0 ):
        #    for x in range(NumeroIndividuos):
        #        print(Poblacion[x].getFitness()   )


        if (generaciones>1000):
            StopCriteria = False
            print(ErrorMin,"Salida por generaciones")
            print(generaciones, " generaciones")
        if ErrorMin[0]<0.01:
            StopCriteria = False
            print(ErrorMin, "Salida por Error Minimo Cumplido")
            print(generaciones, " generaciones")

        if ContadorRep >20 :
            StopCriteria = False
            print(ErrorMin,"Salida por Minimo Estancado")
            print(generaciones, " generaciones")

        if ErrorMin == ErrorAnter:
            ContadorRep = ContadorRep +1
        else:
            ContadorRep = 0

        ErrorAnter = ErrorMin
        
# primero debemos cruzar luego mutar y al ultimo vamos a comparar y tomar al peor padre y cambiarlo con el hijo o dejarlo


        

    



NumEntradas = detEntradas() # primero se determina cuantas entradas tiene el archivo
InitPoblacion() # segundo se inicializa con la primera generacion
AlgoritmoGenetico()

#print(int(np.round(4.0))& int(np.round(5.0)))

#print(Poblacion[0].getPeBiOcultaUnoxUno(1))
