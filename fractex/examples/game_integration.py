# Полная интеграция с Unity/Unreal
import sys
from pathlib import Path
import math
from typing import Tuple, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fractex.dynamic_textures_3d import StreamingDynamicTextures, DynamicTextureType


class GameDynamicTexturesManager:
    def __init__(self, world_size: Tuple[int, int, int]):
        # Разделяем мир на чанки
        self.chunk_size = (32, 32, 32)
        self.num_chunks = (
            world_size[0] // self.chunk_size[0],
            world_size[1] // self.chunk_size[1],
            world_size[2] // self.chunk_size[2],
        )
        
        self.streamer = StreamingDynamicTextures(
            chunk_size=self.chunk_size,
            max_active_chunks=8
        )
        
        self.render_cache = {}
    
    def update(self, dt: float, player_position: Tuple[float, float, float]):
        visible_chunks = self._get_visible_chunks(player_position)
        
        for chunk_coords in visible_chunks:
            distance = self._distance_to_chunk(player_position, chunk_coords)
            priority = 1.0 / (distance + 1.0)
            
            state = self.streamer.request_chunk(
                chunk_coords,
                DynamicTextureType.WATER_FLOW,
                priority
            )
            
            if state:
                self._render_or_update_chunk(chunk_coords, state)
        
        try:
            self.streamer.update_all(dt)
        except Exception as exc:
            print(f"Simulation update skipped: {exc}")
    
    def _get_visible_chunks(self, player_position: Tuple[float, float, float]) -> List[Tuple[int, int, int]]:
        """Получение чанков в поле зрения игрока (упрощенно)"""
        px, py, pz = player_position
        cx = int(px // self.chunk_size[0])
        cy = int(py // self.chunk_size[1])
        cz = int(pz // self.chunk_size[2])
        
        visible = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    if 0 <= nx < self.num_chunks[0] and 0 <= ny < self.num_chunks[1] and 0 <= nz < self.num_chunks[2]:
                        visible.append((nx, ny, nz))
        
        return visible
    
    def _distance_to_chunk(self, player_position: Tuple[float, float, float], chunk_coords: Tuple[int, int, int]) -> float:
        px, py, pz = player_position
        cx, cy, cz = chunk_coords
        center = (
            (cx + 0.5) * self.chunk_size[0],
            (cy + 0.5) * self.chunk_size[1],
            (cz + 0.5) * self.chunk_size[2],
        )
        return math.dist(player_position, center)
    
    def _render_or_update_chunk(self, chunk_coords: Tuple[int, int, int], state):
        self.render_cache[chunk_coords] = state.data.mean()


if __name__ == "__main__":
    manager = GameDynamicTexturesManager(world_size=(96, 96, 96))
    manager.update(dt=0.016, player_position=(48.0, 48.0, 48.0))
    print("Render cache size:", len(manager.render_cache))
