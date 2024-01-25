import sys
sys.path.append('../Utilities') 
import utils
k_sizes = [3000, 5000, 10000, 30000, 50000, 100000, 130000]
nombre_metrica = 'MSE'

metrica_modelo1 = [0.440957201154608,
                   0.324772753885814,
                   0.311555277228355,
                   0.357496356447538,
                   0.329103043913841,
                   0.258324796712399,
                   0.256098675319782]
metrica_modelo2  = [0.216403314941808,
                        0.270002668723464,
                        0.196926709678438,
                        0.233100004890498,
                        0.176112656847547,
                        0.213731021165848,
                        0.193042456966265]
metrica_modelo3  = [0.0388010520684092,
           0.0370032492792234,
           0.0260805724750436,
           0.0396586707297792,
           0.0408545807599069,
           0.0337396560916825,
           0.041145069861434
           ]



utils.guardar_grafica_lineas_multiples_2(metrica_modelo1, metrica_modelo2, metrica_modelo3, k_sizes, nombre_metrica, nombre_modelo1='GCN', nombre_modelo2='GAT', nombre_modelo3='SchNet', nombre_archivo='lines_gnn_MSE.png', direccion='.')


nombre_metrica = 'MAE'

metrica_modelo1 = [0.292579226980084,
                   0.182131003115385,
                   0.155946201741695,
                   0.21944403342406,
                   0.16090961278677,
                   0.125747327259183,
                   0.118415715056543]

metrica_modelo2 = [0.0845592455252221,
                        0.106490095262416,
                        0.0770705947800288,
                        0.0950389540496659,
                        0.0822380284954327,
                        0.0886512351661921,
                        0.0821485973729639]
metrica_modelo3 = [0.00248825875429535,
                  0.00218143788697489,
                  0.00108292344809761,
                  0.0025720312761668,
                  0.00271380713846261,
                  0.00161787914943748,
                  0.00272810748872218]

utils.guardar_grafica_lineas_multiples_2(metrica_modelo1, metrica_modelo2, metrica_modelo3, k_sizes, nombre_metrica, nombre_modelo1='GCN', nombre_modelo2='GAT', nombre_modelo3='SchNet', nombre_archivo='lines_gnn_MAE.png', direccion='.')

nombre_metrica = 'R²'
metrica_modelo1 = [0.580754229896947,
                        0.747269617186652,
                        0.773419684886932,
                        0.67564216486613,
                        0.764088813781738,
                        0.815737518167496,
                        0.824494688877693]
metrica_modelo1 =  [0.885233000705117,
                        0.847030265256763,
                        0.896538195155916,
                        0.877239884531244,
                        0.8909652038885,
                        0.885011405086517,
                        0.891507505857343]
metrica_modelo3 = [0.996630596487146,
                   0.997233010828495,
                   0.998521902258434,
                   0.996601165609157,
                   0.996299259007548,
                   0.99774612320794,
                   0.996273339527853]

utils.guardar_grafica_lineas_multiples_2(metrica_modelo1, metrica_modelo2, metrica_modelo3, k_sizes, nombre_metrica, nombre_modelo1='GCN', nombre_modelo2='GAT', nombre_modelo3='SchNet', nombre_archivo='lines_gnn_SchNet.png', direccion='.')