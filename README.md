# simulacion_poisson
En este proyecto se simula un proceso de Poisson (llegada de automóviles a un estacionamiento) a partir de arribos conocidos en un periodo de tiempo.

Se estima el parámetro lambda, y con el test Chi2 se decide si el valor obtenido es un valor aceptable.

Con el parámetro lambda estimado se simulan 1000 arribos al estacionamiento, generando estos arribos con el generador de valores aleatorios Xorshift.

Se comparan la probabilidad de tres eventos (A, B y C), calculadas de manera teórica y de manera experimental, y se las contrasta calculando el error
relativo porcentual:
  A: "Llega un vehículo antes de los 10 minutos"
  B: "Llegan 10 o más vehículos antes de la hora"
  C: "Llegan 750 vehículos o menos antes de las 72 horas"

