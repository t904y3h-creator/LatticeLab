import sys

import numpy as np


def compare_simulations(file1, file2, rtol=1e-05, atol=1e-08):
    data1 = np.fromfile(file1, dtype=np.float32)
    data2 = np.fromfile(file2, dtype=np.float32)

    if data1.shape != data2.shape:
        print(f"Ошибка: Размеры файлов не совпадают! {data1.shape} vs {data2.shape}")
        return

    diff = np.abs(data1 - data2)

    is_equal = np.allclose(data1, data2, rtol=rtol, atol=atol)

    if is_equal:
        print("✅ Результаты идентичны (в пределах погрешности)")
        print(f"Максимальное отклонение: {np.max(diff)}")
    else:
        mismatch_indices = np.where(~np.isclose(data1, data2, rtol=rtol, atol=atol))[0]
        print("❌ Обнаружены расхождения!")
        print(
            f"Количество несовпадающих элементов: {len(mismatch_indices)} из {len(data1)}"
        )
        print(f"Максимальная разница: {np.max(diff)}")

        # Вывод первых нескольких расхождений для анализа
        print("\nПервые 5 расхождений:")
        for idx in mismatch_indices[:5]:
            print(
                f"Index [{idx}]: V1 = {data1[idx]}, V2 = {data2[idx]}, Diff = {diff[idx]}"
            )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python compare.py <file1> <file2>")
    else:
        compare_simulations(sys.argv[1], sys.argv[2])
