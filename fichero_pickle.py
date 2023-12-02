# -*- coding: utf-8 -*-

from VersionFinal import DatosP, DatosL, RecomendacionSimple, RecomendacionColaborativa, RecomendacionContenido
import pickle

ficherosL = ["ratings.csv", "books.csv"]
ficherosP = ["ratings.csv", "movies.csv"]

d1 = DatosL()
d1.leer_datos(ficherosL[0], ficherosL[1])
d1.crear_tabla()
            
d2 = DatosP()
d2.leer_datos(ficherosP[0], ficherosP[1])
d2.crear_tabla()

rs1 = RecomendacionSimple(d1)

with open("BookSimple.dat", 'wb') as fichero:
    pickle.dump(rs1, fichero)

rs2 = RecomendacionSimple(d2)

with open("MovieSimple.dat", 'wb') as fichero:
    pickle.dump(rs2, fichero)

rcol1 = RecomendacionColaborativa(d1)

with open("BookCol.dat", 'wb') as fichero:
    pickle.dump(rs2, fichero)

rcol2 = RecomendacionColaborativa(d2)

with open("MovieCol.dat", 'wb') as fichero:
    pickle.dump(rs2, fichero)

rcon1 = RecomendacionContenido(d1)

with open("BookCont.dat", 'wb') as fichero:
    pickle.dump(rs2, fichero)

rcon2 = RecomendacionContenido(d2)

with open("MovieCont.dat", 'wb') as fichero:
    pickle.dump(rs2, fichero)

