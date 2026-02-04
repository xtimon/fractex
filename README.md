## Fractex

Fractex — библиотека для генерации процедурных фрактальных текстур и шума
с бесконечной детализацией.

Репозиторий: https://github.com/xtimon/fractex.git

## Установка

```bash
pip install fractex
```

## Быстрый старт

```python
from fractex import FractalParams, FractalGenerator, InfiniteTexture

params = FractalParams(seed=42, detail_level=2.0)
generator = FractalGenerator(params)

clouds = InfiniteTexture(generator, "clouds")
tile = clouds.generate_tile(0, 0, 256, 256, zoom=1.0)
print(tile.shape, tile.min(), tile.max())
```

## CLI

```bash
fractex --list
fractex splash --preset lava
fractex terrain --interactive --fps 30 --scale 1.0
```

## Интерактивный API

```python
import fractex as fx

print(fx.list_examples())
fx.run_example("splash", ["--preset", "marble", "--fps", "30"])
```

## Возможности

- Фрактальный шум (fBm), симплекс‑шум
- Бесконечная детализация через адаптивные октавы
- Пресеты текстур: облака, мрамор, дерево, лава, вода и др.
- Интерактивные примеры с адаптацией качества к FPS

## Лицензия

MIT License. См. файл `LICENSE`.
