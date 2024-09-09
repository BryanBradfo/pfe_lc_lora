import numpy as np
import zlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import zlib
import time
from collections import defaultdict

def group_values(data):
    # Cette fonction est appelée par chaque processus
    δt, δt_exp, δt_sign = data
    groups = defaultdict(list)
    for value, exp, sign in zip(δt, δt_exp, δt_sign):
        groups[(exp, sign)].append(value)
    return groups

def compress_data_optimized(δt, num_bits=3, threshhold=True):
    # Obtenir les exposants et les signes en utilisant les fonctions Numpy
    δt_exp = np.frexp(δt)[1]
    δt_sign = (np.sign(δt) < 0).astype(int)

    # Répartition de la tâche de groupement
    num_processes = cpu_count()
    chunk_size = len(δt) // num_processes
    data_chunks = [(δt[i:i + chunk_size], δt_exp[i:i + chunk_size], δt_sign[i:i + chunk_size])
                   for i in range(0, len(δt), chunk_size)]

    with Pool(num_processes) as pool:
        results = pool.map(group_values, data_chunks)

    # Combinaison des résultats de tous les processus
    combined_groups = defaultdict(list)
    for result in results:
        for key, values in result.items():
            combined_groups[key].extend(values)

    # Calcul des moyennes
    bucket_list = [(np.mean(values), np.array(indices)) for (indices, values) in combined_groups.items()]

    if threshhold:
        allowed_buckets = (1 << num_bits) - 1
        bucket_list = sorted(bucket_list, key=lambda x: abs(x[0]), reverse=True)[:allowed_buckets]

    # Création du nouveau delta
    new_δt = np.zeros_like(δt, dtype=np.float32)
    for val, indices in bucket_list:
        new_δt[indices] = val

    # Compression du nouveau delta
    compressed_δt = zlib.compress(new_δt.tobytes())

    return compressed_δt, new_δt

def compress_data(δt, num_bits=3, threshhold=True):
    _, δt_exp = np.frexp(δt)
    δt_sign = (np.sign(δt) < 0).astype(int)
    groups = defaultdict(list)
    for value, exp, sign in zip(δt, δt_exp, δt_sign):
        groups[(exp, sign)].append(value)
    bucket_list = [(np.mean(values), np.array(indices)) for (indices, values) in groups.items()]
    if threshhold:
        allowed_buckets = (1 << num_bits) - 1
        bucket_list = sorted(bucket_list, key=lambda x: abs(x[0]), reverse=True)[:allowed_buckets]
    new_δt = np.zeros_like(δt, dtype=np.float32)
    for val, indices in bucket_list:
        new_δt[indices] = val
    compressed_δt = zlib.compress(new_δt)
    return compressed_δt, new_δt

def main():
    # Generate a random data array
    np.random.seed(0)
    δt_test = np.random.randn(100000000)

    # Measure performance of the original function
    start_time = time.time()
    original_compressed_data, original_uncompressed_data = compress_data(δt_test, num_bits=3, threshhold=True)
    original_elapsed_time = time.time() - start_time

    # Measure performance of the optimized function
    start_time = time.time()
    optimized_compressed_data, optimized_uncompressed_data = compress_data_optimized(δt_test, num_bits=3, threshhold=True)
    optimized_elapsed_time = time.time() - start_time

    # Print results
    print("Original function execution time:", original_elapsed_time)
    print("Optimized function execution time:", optimized_elapsed_time)

if __name__ == "__main__":
    main()

# def main():
#     # Générer un tableau de données aléatoires
#     np.random.seed(0)
#     δt_test = np.random.randn(1000000)

#     # Test de la fonction
#     compressed_data, uncompressed_data = compress_data_optimized(δt_test, num_bits=3, threshhold=True)
#     print("Taille des données compressées :", len(compressed_data))
#     print("Forme des données non compressées :", uncompressed_data.shape)

#     # Décompresser pour vérifier l'intégrité
#     decompressed_data = np.frombuffer(zlib.decompress(compressed_data), dtype=np.float32)
#     np.testing.assert_array_almost_equal(decompressed_data, uncompressed_data, decimal=5)
#     print("Test de validation de l'intégrité passé!")

#     # Mesurer le temps d'exécution
#     import time
#     start_time = time.time()
#     compress_data_optimized(δt_test, num_bits=3, threshhold=True)
#     elapsed_time = time.time() - start_time
#     print("Temps d'exécution :", elapsed_time)

# if __name__ == "__main__":
#     main()
