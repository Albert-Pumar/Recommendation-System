# -*- coding: utf-8 -*-
"""

--- PROYECTO SISTEMA DE RECOMENDACIÓN ---

- NOMS: Albert Pumar

"""

from dataclasses import dataclass, field
from typing import List, Dict
from abc import ABCMeta, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv

@dataclass
class Recomendacion(metaclass = ABCMeta):
    _datos: object
    
    @abstractmethod
    def recomienda(self):
        raise NotImplementedError()
    @abstractmethod
    def evalua(self):
        raise NotImplementedError()
    @abstractmethod
    def prediccion(self):
        raise NotImplementedError()
    
    def comparacion(self, predicciones: list, valoraciones_reales: list, umbral: int, N: int):
        # CÁLCULO MAE
        num = 0
        for i in range(len(predicciones)):
            num += abs(predicciones[i] - valoraciones_reales[i])
        
        mae = num / len(predicciones)
        
        #Reordenamos las listas
        predicciones, valoraciones_reales = zip(*sorted(zip(predicciones, valoraciones_reales), reverse = True))
        
        # CÁLCULO PRECISION
        conj = valoraciones_reales[:N]
        cont = 0
        for valoracion in conj:
            if valoracion >= umbral:
                cont += 1
        precision = cont / N
        
        # CÁLCULO RECALL
        num = 0
        den = 0
        valoraciones = []
        for i in range(len(valoraciones_reales)):
            if valoraciones_reales[i] >= umbral:
                valoraciones.append((i, valoraciones_reales[i]))
                if i < N:
                    num += 1
                    den += 1
                else:
                    den += 1
        try:
            recall = num / den
        except:
            recall = 0
        
        return conj, valoraciones, mae, precision, recall

@dataclass
class RecomendacionSimple(Recomendacion):
    def __init__(self, datos):
        super().__init__(datos)
        
    def recomienda(self, min_votos: int, user: int):
        scores = [] # Lista de cada score de cada item
        pos = self._datos.dict_users[user]
        
        valoraciones_user = list(self._datos.tabla[:, pos]) #Lista de las valoraciones del usuario
        
        no_val_antes_el = [] #0,4
        for i in range(len(valoraciones_user)):
            if valoraciones_user[i] == 0:
                no_val_antes_el.append(i) #Guardamos la posición de los items no valorados antes de eliminar algun ítem
        
        num_votos, avg_items, avg_global = self.prediccion(valoraciones_user, self._datos.tabla, min_votos)
        
        #GUARDAMOS LAS POSICIONES DE LOS ITEMS NO VALORADOS
        no_valoradas = [] #0,4
        for i in range(len(num_votos)):
            if valoraciones_user[i] == 0:
                no_valoradas.append(i) #Guardamos la posición de los items no valorados
        
        #CALCULAMOS scores
        for i in range(len(no_valoradas)):
            score = (num_votos[no_valoradas[i]] / (num_votos[no_valoradas[i]] + min_votos) * avg_items[no_valoradas[i]]) + (min_votos / (num_votos[no_valoradas[i]] + min_votos) * avg_global)
            idd = self._datos.list_items[no_val_antes_el[i]]
            scores.append((score, str("ID: "+str(idd))+" - Título: "+str(self._datos.items[idd].titulo)+" - Características: "+str(self._datos.items[idd].caracteristicas)))
        
        bests_rec = sorted(scores, reverse = True)[:5]
        
        return bests_rec
    
    def evalua(self, user: int, N: int, umbral: int, min_votos: int): #!!!
        param = 0.2 # EL 20% A TEST Y EL 80% A TRAIN
        x = round(len(self._datos.tabla) * param)
        test = self._datos.tabla[:x,:] #Cogemos el primer 20% de filas como test
        train = self._datos.tabla[x:,:] #Cogemos el 80% restante como train
        
        pos = self._datos.dict_users[user]
        valoraciones_reales_user = list(self._datos.tabla[:, pos])
        
        no_valoradas = [0] * len(test)
        valoraciones_user = valoraciones_reales_user[:len(train)] + no_valoradas
        scores = []
        
        num_votos, avg_items, avg_global = self.prediccion(valoraciones_user, test, min_votos)
        
        suma = 0
        numero = 0
        for i in range(len(train)):
            for j in range(len(train[0])):
                if train[i][j] != 0:
                    suma += train[i][j]
                    numero += 1
        avg_global = suma / numero
        
        #CALCULAMOS scores
        for posicion in valoraciones_user:
            score = (num_votos[posicion] / (num_votos[posicion] + min_votos) * avg_items[posicion]) + (min_votos / (num_votos[posicion] + min_votos) * avg_global)
            scores.append((score, self._datos.list_items[posicion]))
        
        #CÁLCULO DE LAS MEDIDAS DE COMPARACIÓN
        conj, valoraciones, mae, precision, recall = self.comparacion(scores, valoraciones_reales_user, umbral, N)
        
        N_mejores_predicciones = []
        for i in range(len(conj)):
            idd = self._datos.list_items[i]
            N_mejores_predicciones.append((idd, conj[i]))
        
        valoraciones_mayores_al_umbral = []
        for valoracion in valoraciones:
            idd = self._datos.list_items[valoracion[0]]
            valoraciones_mayores_al_umbral.append((idd, valoracion[1]))
        
        return N_mejores_predicciones, valoraciones_mayores_al_umbral, mae, precision, recall
        
    def prediccion(self, valoraciones_user, tabla = None, min_votos = None, test = None):
        avg_items = [] # Lista con la media de valoraciones de cada item
        num_votos = [] # Lista del numero de valoraciones de cada item
        sum_global = 0 # Suma de todas las valoraciones
        
        #CALCULAMOS avg_item
        for i in range(len(tabla)):
            votos = 0 # Numero de iteraciones del item de la iteracióm actual
            avg_item = 0 # Valor de la media de valoraciones del item 
            for j in range(len(tabla[0])):
                if tabla[i][j] != 0:
                    votos += 1
                    avg_item += tabla[i][j]
    
            if votos >= min_votos:
                num_votos.append(votos) 
                sum_global += avg_item
                avg_item = avg_item / votos
                avg_items.append(avg_item)
            else:
                valoraciones_user[i] = -1
                
        valoraciones_user = [i for i in valoraciones_user if i != -1]
                
        #CALCULAMOS avg_global
        avg_global = sum_global / sum(num_votos)
        
        return num_votos, avg_items, avg_global
    
    def comparacion(self, predicciones: list, valoraciones_reales: list, umbral: int, N: int):
        conj, valoraciones_mayores_al_umbral, mae, precision, recall = super().comparacion(predicciones, valoraciones_reales, umbral, N)
        
        return conj, valoraciones_mayores_al_umbral, mae, precision, recall

@dataclass      
class RecomendacionColaborativa(Recomendacion):
    def __init__(self, datos):
        super().__init__(datos)
    
    def recomienda(self, user: int):
        pos = self._datos.dict_users[user]
        valoraciones_user = list(self._datos.tabla[:, pos])
        
        self._datos.tabla = np.transpose(self._datos.tabla)
        
        no_valoradas = [] #0,4
        for i in range(len(valoraciones_user)):
            if valoraciones_user[i] == 0:
                no_valoradas.append(i) #Guardamos la posición de los items no valorados
        
        media_val_user = sum(valoraciones_user) / (len(valoraciones_user) - len(no_valoradas))
        
        valoraciones = self.prediccion(valoraciones_user, self._datos.tabla, self._datos.tabla, no_valoradas, media_val_user) #Solo cuando se recomienda test = self._datos.tabla

        return valoraciones
        
    def prediccion(self, valoraciones_user, tabla, test, no_valoradas, media_val_user):
        similitudes = []

        #CÁLCULO SIMILITUD
        for i in range(len(tabla)):
            numerador = 0
            den1 = 0 #Primera raíz
            den2 = 0 #Segunda raíz
            for j in range(len(tabla[0])):
                if tabla[i][j] != 0 and valoraciones_user[j] != 0:
                    numerador += tabla[i][j] * valoraciones_user[j]
                    den1 += tabla[i][j] * tabla[i][j]
                    den2 += valoraciones_user[j]*valoraciones_user[j]
            
            if den1 != 0 and den2 != 0:
                similitud = numerador / ((den1)**(1/2) * (den2)**(1/2))
                similitudes.append((similitud, i))

        best_users = sorted(similitudes, reverse = True)[:5]
        print(best_users)
        #CÁLCULO PUNTUACIÓN FINAL
        best_rec = []
        scores = []
        
        #Cálculo medias best_users
        medias = []
        for usuario in best_users:
            pos_usuario = usuario[1]
            user = tabla[pos_usuario] #fila de la tabla, conjunto valoraciones de un usuario
            count = 0
            suma = 0
            for j in range(len(user)):
                if user[j] != 0:
                    count += 1
                    suma += user[j]
                    
            medias.append(suma / count)
        
        #Cálculo Scores
        for i in range(len(no_valoradas)):
            num = 0
            den = 0
            score = media_val_user
            for j in range(len(best_users)):
                usuario = test[best_users[j][1]]
                num += best_users[j][0] * (usuario[no_valoradas[i]] - medias[j])
                den += best_users[j][0]
            
            #if den != 0: #Mismo caso que el mencionado antes, hay usuarios que no han puntuado ningún ítem de train o test
            score += num / den
            scores.append((score, "P"+str(no_valoradas[i]+1)))
        
        best_rec = sorted(scores, reverse = True)[:5]
        
        return best_rec
    
    def evalua(self, user: int, umbral: int, N: int): # !!!
        self._datos.tabla = np.transpose(self._datos.tabla)
        param = 0.2 # EL 20% A TEST Y EL 80% A TRAIN
        x = round(len(self._datos.tabla) * param)
        test = self._datos.tabla[:,:x] #Cogemos el primer 20% de filas como test
        train = self._datos.tabla[:,x:] #Cogemos el 80% restante como train
        
        pos = self._datos.dict_users[user]
        valoraciones_reales_user = list(self._datos.tabla[pos])
        
        no_valoradas = [0] * len(test[0])
        valoraciones_user = valoraciones_reales_user[:len(train[0])] + no_valoradas

        media_val_user = sum(valoraciones_user) / (len(valoraciones_user) - len(no_valoradas))
        
        valoraciones = self.prediccion(valoraciones_user, train, test, no_valoradas, media_val_user) #SOlo cuando se recomienda test = self._datos.tabla
        
        #CÁLCULO DE LES MEDIDAS DE COMPARACIÓN
        val = [x[0] for x in valoraciones]
        
        conj, valoraciones, mae, precision, recall = self.comparaciom(val, valoraciones_reales_user, umbral, N)
        
        N_mejores_predicciones = []
        for i in range(len(conj)):
            idd = self._datos.list_items[i]
            N_mejores_predicciones.append((idd, conj[i]))
        
        valoraciones_mayores_al_umbral = []
        for valoracion in valoraciones:
            idd = self._datos.list_items[valoracion[0]]
            valoraciones_mayores_al_umbral.append((idd, valoracion[1]))
        
        return N_mejores_predicciones, valoraciones_mayores_al_umbral, mae, precision, recall
    
    def comparacion(self, predicciones: list, valoraciones_reales: list, umbral: int, N: int):
        conj, valoraciones_mayores_al_umbral, mae, precision, recall = super().comparacion(predicciones, valoraciones_reales, umbral, N)
        
        return conj, valoraciones_mayores_al_umbral, mae, precision, recall

@dataclass
class RecomendacionContenido(Recomendacion):
    def __init__(self, datos):
        super().__init__(datos)
    
    def recomienda(self, user:int):
        pos = self._datos.dict_users[user]
        valoraciones_user = list(self._datos.tabla[:, pos])
        
        #1. Obtener representación ítems
        item_features = []
        for i in range(len(self._datos.tabla)):
            id_item = self._datos.list_items[i]
            item_features.append(self._datos.items[id_item].caracteristicas)
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(item_features).toarray()
        
        similitudes = self.prediccion(valoraciones_user, user, tfidf_matrix, tfidf_matrix)
        
        s = sorted(similitudes, reverse = True)[:5]
        
        return s
    
    def prediccion(self, valoraciones_user, user, tfidf_matrix_train, tfidf_matrix_test):        
        #2. Calcular perfil usuario
        profile_user = []
        den = sum(valoraciones_user)
        for i in range(len(tfidf_matrix_train)):
            num = valoraciones_user[i] * sum(tfidf_matrix_train[i])
            profile_user.append(num/den)
        
        #3. Calcular distancia cosino
        similitudes = []
        for i in range(len(tfidf_matrix_test)):
            numerador = 0
            den1 = 0 #Primera raíz
            den2 = 0 #Segunda raíz
            for j in range(len(tfidf_matrix_test[0])):
                if tfidf_matrix_test[i][j] != 0 and profile_user[i] != 0:
                    numerador += tfidf_matrix_test[i][j] * profile_user[i]
                    den1 += tfidf_matrix_test[i][j] * tfidf_matrix_test[i][j]
                    den2 += profile_user[i] * profile_user[i]
            try:
                similitud = numerador / ((den1)**(1/2) * (den2)**(1/2))
                #4. Calcular puntuación final de cada ítem
                puntuacion_max = len(tfidf_matrix_test[0]) #Consideramos que la puntuación máxima es aquella donde todos 
                #los elementos del vector son iguales a 1, coincidencia total, por tanto la suma de todos los 1 del vector
                similitudes.append((similitud * puntuacion_max, i))
            except:
                similitudes.append((0, i))
        
        return similitudes
        
    def evalua(self, user: int, umbral: int, N: int): # !!!
        param = 0.2 # EL 20% A TEST Y EL 80% A TRAIN
        x = round(len(self._datos.tabla) * param)
        test = self._datos.tabla[:,:x] #Cogemos el primer 20% de filas como test
        train = self._datos.tabla[:,x:] #Cogemos el 80% restante como train
        
        pos = self._datos.dict_users[user]
        valoraciones_reales_user = list(self._datos.tabla[:, pos])
        
        no_valoradas = [0] * len(test)
        valoraciones_user = valoraciones_reales_user[:len(train)] + no_valoradas
        
        #ITEM FEATURES DE LOS ITEMS DE TRAIN
        item_features_train = []
        for i in range(len(train)):
            id_item = self._datos.list_items[i]
            item_features_train.append(self._datos.items[id_item].caracteristicas)
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix_train = tfidf.fit_transform(item_features_train).toarray()
        
        #ITEM FEATURES DE LOS ITEMS DE TEST
        item_features_test = []
        for i in range(len(test)):
            id_item = self._datos.list_items[i]
            item_features_test.append(self._datos.items[id_item].caracteristicas)
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix_test = tfidf.fit_transform(item_features_test).toarray()
            
        similitudes = self.prediccion(valoraciones_user, user, tfidf_matrix_train, tfidf_matrix_test)
        
        #CÁLCULO DE LAS MEDIDAS DE COMPARACIÓN
        valoraciones = [x[0] for x in similitudes]
        
        conj, valoraciones, mae, precision, recall = self.comparacion(valoraciones, valoraciones_reales_user, umbral, N)
        
        N_mejores_predicciones = []
        for i in range(len(conj)):
            idd = self._datos.list_items[i]
            N_mejores_predicciones.append((idd, conj[i]))
        
        valoraciones_mayores_al_umbral = []
        for valoracion in valoraciones:
            idd = self._datos.list_items[valoracion[0]]
            valoraciones_mayores_al_umbral.append((idd, valoracion[1]))
        
        return N_mejores_predicciones, valoraciones_mayores_al_umbral, mae, precision, recall
        
    def comparacion(self, predicciones: list, valoraciones_reales: list, umbral: int, N: int):
        conj, valoraciones_mayores_al_umbral, mae, precision, recall = super().comparacion(predicciones, valoraciones_reales, umbral, N)
        
        return conj, valoraciones_mayores_al_umbral, mae, precision, recall

@dataclass
class Item(metaclass = ABCMeta):
    _id: int
    _titulo: str
    _caracteristicas: str #Para hacer la recomendación basada en contenido en libros son los autores y en las pelis, los géneros
    
    @property
    def idd(self):
        return self._id
    @property
    def titulo(self):
        return self._titulo
    @property
    def caracteristicas(self):
        return self._caracteristicas
    
class Pelicula(Item):
    def __init__(self, identificador: str, titulo: str, caracteristicas: str):
        super().__init__(identificador, titulo, caracteristicas)
        
class Libro(Item):
    def __init__(self, identificador: str, titulo: str, caracteristicas: str):
        super().__init__(identificador, titulo, caracteristicas)

@dataclass
class Usuario():
    _id: int
    _predicciones_test: List[int] = field(default_factory = list)
            
    @property
    def predicciones_test(self):
        return self._predicciones_test
    @predicciones_test.setter
    def predicciones_test(self, value):
        self._predicciones_test = value

@dataclass
class Datos(metaclass = ABCMeta):
    _items: Dict[int, object] = field(default_factory = dict) #Nos sirve para obtener un objeto Item a partir de sus id
    _list_items: List[int] = field(default_factory = list) #Nos sirve para obtener una id dada una posición (saber a qué item corresponde una valoración)
    _dict_users: Dict[int, int] = field(default_factory = dict) #Nos sirve para saber a qué fila se encuentra un id de user
    _usuarios: Dict[int, object] = field(default_factory = dict) #Nos sirve para obtener un objeto Usuario a partir de sus id
    _puntuaciones: Dict[str, list] = field(default_factory = dict)
    #puntuaciones cuando sea Libro, será: {ID_libro: lista valoraciones de los usuarios a este libro}
    #puntuaciones cuando sea Película, será: {ID_usuario: lista valoraciones del usuario a las películas}
    _tabla: np.ndarray = np.zeros((1,1), dtype = "i")
    
    @abstractmethod
    def leer_datos(self, fitxer: str):
        raise NotImplementedError()
    @abstractmethod
    def crear_tabla(self):
        raise NotImplementedError()
    
    @property
    def usuarios(self):
        return self._usuarios
    @property
    def items(self):
        return self._items
    @property
    def puntuaciones(self):
        return self._puntuaciones
    @property
    def tabla(self):
        return self._tabla
    @tabla.setter
    def tabla(self, value):
        self._tabla = value
    @property
    def dict_users(self):
        return self._dict_users
    @property
    def list_items(self):
        return self._list_items
    
@dataclass
class DatosP(Datos):
    def __init__(self):
        super().__init__()
        
    def leer_datos(self, fichero_r = "ratings.csv", fichero_m = "movies.csv"):
        #LEEMOS FICHERO movies.csv
        with open("movies\\"+fichero_m, "r", encoding = "utf8") as file_m:
            csvreader = csv.reader(file_m, delimiter = ";")
            next(csvreader) #Saltamos el header
            for line in csvreader:
                p = Pelicula(int(line[0]), line[1], line[2]) # Añadimos los datos de las películes
                self._items[int(line[0])] = p #Guardamos el objeto y la columna que ocupa
                self._list_items.append(int(line[0]))
        
        #LEEMOS FICHERO ratings.csv
        with open("movies\\"+fichero_r, "r", encoding = "utf8") as file_r:
            csvreader = csv.reader(file_r)
            next(csvreader)
            count = 0
            for line in csvreader:
                if int(line[0]) not in self._puntuaciones:
                    line[2] = float(line[2]) #Corrección de tipo de datos, me daba error al acceder a los indices del nparray
                    self._puntuaciones[int(line[0])] = [(int(line[1]), int(line[2]))] #guardamos id de la película, posicion en nparray y la valoración
                    self._usuarios[int(line[0])] = Usuario(int(line[0]))
                    self._dict_users[int(line[0])] = count
                else:
                    line[2] = float(line[2])
                    self._puntuaciones[int(line[0])].append((int(line[1]), int(line[2])))
            count += 1
    
    def crear_tabla(self):
        self._tabla = np.zeros((len(self._puntuaciones), len(self._items)), dtype = "i")
        
        for elemento in self._puntuaciones:
            for i in range(len(self._puntuaciones[elemento])):
                self._tabla[elemento - 1][i] = self._puntuaciones[elemento][i][1] #En las películas que haya valorado el usuario, cambiamos el 0 por la valoración

        self._tabla = np.transpose(self._tabla) #Para hacer los cáculos de la recomendación, mejor con un item por fila
        
@dataclass
class DatosL(Datos):
    def __init__(self):
        super().__init__()
        
    def leer_datos(self, fichero_r = "ratings.csv", fichero_b = "books.csv"):
        #LEEMOS FICHERO ratings.csv
        with open("books\\"+fichero_r, "r", encoding = "utf8") as file:
            csvreader = csv.reader(file)
            next(csvreader)
            count = 0
            for line in csvreader:
                if int(line[0]) not in self._puntuaciones:
                    self._puntuaciones[int(line[0])] = [(int(line[1]), int(line[2]))]
                else:
                    self._puntuaciones[int(line[0])].append((int(line[1]), int(line[2])))
                    
                if int(line[1]) not in self._usuarios:
                    self._usuarios[int(line[1])] = Usuario(int(line[1]))
                    self._dict_users[int(line[0])] = count
                    
                count += 1

        #Añadimos datos a los objetos Libro, leemos fitxer books.csv
        with open("books\\"+fichero_b, "r", encoding = "utf8") as file:
            csvreader = csv.reader(file, delimiter = ";")
            next(csvreader)
            for line in csvreader:
                l = Libro(int(line[0]), line[2], line[1])
                self._items[int(line[0])] = l
                self._list_items.append(int(line[0]))
    
    def crear_tabla(self):
        self._tabla = np.zeros((len(self._puntuaciones), len(self._items) - 1), dtype = "i")
        
        for elemento in self._puntuaciones:
            for i in range(len(self._puntuaciones[elemento])):
                self._tabla[elemento - 1][i] = self._puntuaciones[elemento][i][1] #el id del usuario se corresponde a la columna del array
                    
    
    