import sys
import subprocess


def main():
    if len(sys.argv) < 4:
        print("Uso: python barrido.py <k_minima> <k_maxima> <p_maxima>")
        sys.exit(1)

    k_min = int(sys.argv[1])
    k_max = int(sys.argv[2])
    p_max = int(sys.argv[3])

    archivo_datos = "iris.csv"
    archivo_config = "configuration.json"
    pesos = ['uniform', 'distance']

    print("Iniciando barrido de hiperparámetros...")

    # Bucle para probar k=1, 3, 5... (saltando de 2 en 2 para valores impares)
    for k in range(k_min, k_max + 1, 2):
        for p in range(1, p_max + 1):
            for w in pesos:
                # Ejecutamos kNN.py como un subproceso
                comando = [
                    "python", "kNN.py",
                    archivo_datos,
                    str(k),
                    w,
                    str(p),
                    "-c", archivo_config
                ]
                subprocess.run(comando)

    print("\nBarrido completado. Revisa el archivo 'resultados_knn.csv' y los modelos .sav generados.")


if __name__ == "__main__":
    main()