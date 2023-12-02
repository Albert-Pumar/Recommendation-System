# -*- coding: utf-8 -*-
"""

--- PROYECTO PROGRAMACIÓN AVANZADA 2021-2022 ---

- NOMBRE: Albert Pumar 
- NIU: 1597973

"""

from VersionFinal import DatosP, DatosL, RecomendacionSimple, RecomendacionColaborativa, RecomendacionContenido
import pickle

ficherosL = ["ratings.csv", "books.csv"]
ficherosP = ["ratings.csv", "movies.csv"]

init = str(input("Selecciona método de inicialización': (Manual/Pickle) "))
assert(init == "Manual" or init == "Pickle"), "ERROR: Método de inicialización introducido incorrecto."

if init == "Manual":
        #PEDIMOS AL USUARIO EL TIPO DE ÍTEM Y DE LA RECOMENDACIÓN QUE QUIERE
        item = str(input("Selecciona Libro o Película: (L/P) ")).upper()
        
        #COMPROBAMOS QUE LOS VALORES ENTRADOS POR TECLADO SEAN CORRECTOS
        assert(item == "L" or item == "P"), "ERROR: Letra introducida incorrecta. Indica con 'L' para escoger libros o 'P' para escoger películas."
        
        if item == "L":
            d = DatosL()
            d.leer_datos(ficherosL[0], ficherosL[1])
            d.crear_tabla()
            
        else:
            d = DatosP()
            d.leer_datos(ficherosP[0], ficherosP[1])
            d.crear_tabla()
else:
    item = str(input("Selecciona Libro o Película: (L/P) "))
    assert(item == "L" or item == "P"), "ERROR: Letra introducida incorrecta. Indica con 'L' para escoger libros o 'P' para escoger películas."
    rec = str(input("Selecciona el tipo de recomendación: (Simple/Colaborativo/Contenido) "))
    assert(rec in ["Simple","Colaborativo", "Contenido"]), "ERROR: Recomendación escogida incorrecta. Selecciona Simple, Colaborativo o Contenido."
    
    if item == "L":
        if rec == "Simple":
            with open("BookSimple.dat", 'rb') as fichero:
                r = pickle.load(fichero)
                
        elif rec == "Colaborativo":
            with open("BookCol.dat", 'rb') as fichero:
                r = pickle.load(fichero)
                
        else:
            with open("BookCont.dat", 'rb') as fichero:
                r = pickle.load(fichero)
        
    else:
        if rec == "Simple":
            with open("MovieSimple.dat", 'rb') as fichero:
                r = pickle.load(fichero)
                
        elif rec == "Colaborativo":
            with open("MovieCol.dat", 'rb') as fichero:
                r = pickle.load(fichero)
                
        else:
            with open("MovieCont.dat", 'rb') as fichero:
                r = pickle.load(fichero)
    

tipo = " "
while tipo:
    tipo = str(input("Selecciona si quieres evaluar, recomendar o salir: (E/R/S) ")).upper()
    assert(tipo in ["E","R","S"]), "ERROR: Selección incorrecta."
    
    if tipo == "S":
        break
    
    if tipo == "R":
        rec = str(input("Selecciona el tipo de recomendación: (Simple/Colaborativo/Contenido) "))
        assert(rec in ["Simple","Colaborativo", "Contenido"]), "ERROR: Recomendación escogida incorrecta. Selecciona Simple, Colaborativo o Contenido."
        
        if rec == "Simple":
            if init == "Manual":
                r = RecomendacionSimple(d)
            un, dos, tres, cuatro, cinco = r.recomienda(3, 1) #user, min_votos
            print("Primer Ítem: "+str(un[1])+" - Valoración: "+str(un[0]))
            print("Segundo Ítem: "+str(dos[1])+" - Valoración: "+str(dos[0]))
            print("Tercer Ítem: "+str(tres[1])+" - Valoración: "+str(tres[0]))
            print("Cuarto Ítem: "+str(cuatro[1])+" - Valoración: "+str(cuatro[0]))
            print("Quinto Ítem: "+str(cinco[1])+" - Valoración: "+str(cinco[0]))
            
        elif rec == "Colaborativo":
            if init == "Manual":
                r = RecomendacionColaborativa(d)
            un, dos, tres, cuatro, cinco = r.recomienda(2) #user
            print("Primer Ítem: "+str(un[1])+" - Valoración: "+str(un[0]))
            print("Segundo Ítem: "+str(dos[1])+" - Valoración: "+str(dos[0]))
            print("Tercer Ítem: "+str(tres[1])+" - Valoración: "+str(tres[0]))
            print("Cuarto Ítem: "+str(cuatro[1])+" - Valoración: "+str(cuatro[0]))
            print("Quinto Ítem: "+str(cinco[1])+" - Valoración: "+str(cinco[0]))
        else:
            if init == "Manual":
                r = RecomendacionContenido(d)
            un, dos, tres, cuatro, cinco = r.recomienda(3) #user
            print("Primer Ítem: "+str(un[1])+" - Valoración: "+str(un[0]))
            print("Segundo Ítem: "+str(dos[1])+" - Valoración: "+str(dos[0]))
            print("Tercer Ítem: "+str(tres[1])+" - Valoración: "+str(tres[0]))
            print("Cuarto Ítem: "+str(cuatro[1])+" - Valoración: "+str(cuatro[0]))
            print("Quinto Ítem: "+str(cinco[1])+" - Valoración: "+str(cinco[0]))
    else:
        rec = str(input("Selecciona el tipo de recomendación: (Simple/Colaborativo/Contenido) "))
        assert(rec in ["Simple","Colaborativo", "Contenido"]), "ERROR: Recomendación escogida incorrecta. Selecciona Simple, Colaborativo o Contenido."
        
        if rec == "Simple":
            if init == "Manual":
                r = RecomendacionSimple(d)
            N, v, mae, p, r = r.evalua(1, 10, 3, 20) #user, N, umbral, min_votos
            print("N MEJORES PREDICCIONES:", N)
            print("VALORACIONES USUARIO:", v)
            print("MAE:", mae)
            print("PRECISION:", p)
            print("RECALL:",r)
            
        elif rec == "Col·laboratiu":
            if init == "Manual":
                r = RecomendacionColaborativa(d)
            N, v, mae, p, r = r.evalua(2, 3, 20) #user, umbral, N
            print("N MEJORES PREDICCIONES:", N)
            print("VALORACIONES USUARIO:", v)
            print("MAE:", mae)
            print("PRECISION:", p)
            print("RECALL:",r)
            
        else:
            if init == "Manual":
                r = RecomendacionContenido(d)
            N, v, mae, p, r = r.evalua(3, 3, 20) #user, umbral, N
            print("N MEJORES PREDICCIONES:", N) 
            print("VALORACIONES USUARIO:", v)
            print("MAE:", mae)
            print("PRECISION:", p)
            print("RECALL:",r)



