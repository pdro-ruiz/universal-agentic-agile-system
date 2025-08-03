#!/usr/bin/env python3
"""
Memory AutoSave System - Automatic Memory Save and Update System
Ensures continuous persistence and automatic synchronization of agent state
"""

import asyncio
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from contextlib import asynccontextmanager
import pickle
import gzip
import hashlib
from enum import Enum

class SavePriority(Enum):
    CRITICAL = 1    # Save immediately
    HIGH = 2        # Save in <5 seconds
    MEDIUM = 3      # Save in <30 seconds
    LOW = 4         # Save in next cycle (60s)

class MemoryType(Enum):
    WORKING = "working"           # Active working memory
    EPISODIC = "episodic"        # Specific event memories
    SEMANTIC = "semantic"         # General knowledge/patterns
    PROCEDURAL = "procedural"     # Skills and procedures
    COLLABORATIVE = "collaborative" # Shared memory between agents

@dataclass
class MemoryChange:
    """Represents a memory change that needs to be saved"""
    agent_id: str
    memory_type: MemoryType
    change_id: str
    timestamp: datetime
    priority: SavePriority
    data: Dict[str, Any]
    checksum: str = ""
    
    def __post_init__(self):
        # Generate checksum to detect changes
        data_str = json.dumps(self.data, sort_keys=True, default=str)
        self.checksum = hashlib.md5(data_str.encode()).hexdigest()

@dataclass
class AutoSaveConfig:
    """AutoSave system configuration"""
    critical_save_interval: float = 0.1    # 100ms para críticos
    high_save_interval: float = 5.0        # 5s para alta prioridad
    medium_save_interval: float = 30.0     # 30s para media prioridad
    low_save_interval: float = 60.0        # 60s para baja prioridad
    
    max_memory_buffer: int = 1000          # Máximo cambios en buffer
    compression_enabled: bool = True        # Comprimir datos al guardar
    incremental_backup: bool = True        # Backups incrementales
    backup_retention_days: int = 30        # Días de retención de backups
    
    # Save strategies
    batch_size: int = 50                   # Cambios por batch
    concurrent_saves: int = 3              # Operaciones simultáneas
    retry_attempts: int = 3                # Reintentos en caso de error

class MemoryAutoSaveOrchestrator:
    """Distributed memory autosave orchestrator"""
    
    def __init__(self, storage_path: str = "./memory_autosave", config: Optional[AutoSaveConfig] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.config = config or AutoSaveConfig()
        
        # Bases de datos especializadas
        self.main_db_path = self.storage_path / "memory_main.db"
        self.changes_db_path = self.storage_path / "memory_changes.db"
        self.backup_db_path = self.storage_path / "memory_backups.db"
        
        # Buffers de cambios por prioridad
        self.change_buffers: Dict[SavePriority, List[MemoryChange]] = {
            priority: [] for priority in SavePriority
        }
        
        # Estado del sistema
        self.save_tasks_active = {}
        self.save_stats = {
            "total_saves": 0,
            "failed_saves": 0,
            "average_save_time": 0.0,
            "last_save_timestamp": None
        }
        
        # Locks para thread safety
        self.buffer_locks = {priority: threading.Lock() for priority in SavePriority}
        self.db_lock = threading.Lock()
        
        # Callbacks para eventos
        self.save_callbacks: Dict[str, Callable] = {}
        
        self._init_databases()
        self._start_autosave_loops()
        
    def _init_databases(self):
        """Inicializa bases de datos optimizadas"""
        
        # Base de datos principal
        with sqlite3.connect(self.main_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory (
                    agent_id TEXT,
                    memory_type TEXT,
                    key TEXT,
                    value BLOB,
                    checksum TEXT,
                    timestamp REAL,
                    version INTEGER,
                    PRIMARY KEY (agent_id, memory_type, key)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_timestamp 
                ON agent_memory(agent_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type 
                ON agent_memory(memory_type, timestamp)
            """)
        
        # Base de datos de cambios (log transaccional)
        with sqlite3.connect(self.changes_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_changes_log (
                    change_id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    memory_type TEXT,
                    timestamp REAL,
                    priority INTEGER,
                    data BLOB,
                    checksum TEXT,
                    saved BOOLEAN DEFAULT FALSE,
                    retry_count INTEGER DEFAULT 0
                )
            """)
        
        # Base de datos de backups incrementales
        with sqlite3.connect(self.backup_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS incremental_backups (
                    backup_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    changes_count INTEGER,
                    compressed_data BLOB,
                    metadata TEXT
                )
            """)
    
    def _start_autosave_loops(self):
        """Starts autosave loops by priority"""
        
        # Loop para cada prioridad con su intervalo específico
        intervals = {
            SavePriority.CRITICAL: self.config.critical_save_interval,
            SavePriority.HIGH: self.config.high_save_interval,
            SavePriority.MEDIUM: self.config.medium_save_interval,
            SavePriority.LOW: self.config.low_save_interval
        }
        
        for priority, interval in intervals.items():
            task = asyncio.create_task(self._autosave_loop(priority, interval))
            self.save_tasks_active[priority] = task
            
        # Loop de backup incremental
        asyncio.create_task(self._incremental_backup_loop())
        
        # Loop de limpieza y optimización
        asyncio.create_task(self._maintenance_loop())
        
        print("AutoSave system started")
        print(f"   ⚡ Crítico: {self.config.critical_save_interval}s")
        print(f"   🔥 Alto: {self.config.high_save_interval}s")
        print(f"   Medium: {self.config.medium_save_interval}s")
        print(f"   💤 Bajo: {self.config.low_save_interval}s")
    
    async def register_memory_change(self, agent_id: str, memory_type: MemoryType, 
                                   data: Dict[str, Any], priority: SavePriority = SavePriority.MEDIUM):
        """Registers memory change for autosave"""
        
        change = MemoryChange(
            agent_id=agent_id,
            memory_type=memory_type,
            change_id=f"{agent_id}_{memory_type.value}_{int(time.time()*1000)}",
            timestamp=datetime.now(),
            priority=priority,
            data=data
        )
        
        # Añadir al buffer correspondiente
        with self.buffer_locks[priority]:
            self.change_buffers[priority].append(change)
            
            # If buffer is full, force save
            if len(self.change_buffers[priority]) >= self.config.max_memory_buffer:
                asyncio.create_task(self._force_save_buffer(priority))
        
        # Log del cambio para auditoria
        await self._log_change(change)
        
        # Ejecutar callbacks si están registrados
        callback_key = f"{agent_id}_{memory_type.value}"
        if callback_key in self.save_callbacks:
            try:
                await self.save_callbacks[callback_key](change)
            except Exception as e:
                print(f"⚠️ Error en callback {callback_key}: {e}")
    
    async def _log_change(self, change: MemoryChange):
        """Registra cambio en log transaccional"""
        
        compressed_data = self._compress_data(change.data) if self.config.compression_enabled else pickle.dumps(change.data)
        
        with sqlite3.connect(self.changes_db_path) as conn:
            conn.execute("""
                INSERT INTO memory_changes_log 
                (change_id, agent_id, memory_type, timestamp, priority, data, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                change.change_id,
                change.agent_id,
                change.memory_type.value,
                change.timestamp.timestamp(),
                change.priority.value,
                compressed_data,
                change.checksum
            ))
    
    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """Comprime datos para almacenamiento eficiente"""
        serialized = pickle.dumps(data)
        return gzip.compress(serialized)
    
    def _decompress_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """Descomprime datos almacenados"""
        serialized = gzip.decompress(compressed_data)
        return pickle.loads(serialized)
    
    async def _autosave_loop(self, priority: SavePriority, interval: float):
        """Main autosave loop for a specific priority"""
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Procesar buffer de esta prioridad
                with self.buffer_locks[priority]:
                    if self.change_buffers[priority]:
                        changes_to_save = self.change_buffers[priority].copy()
                        self.change_buffers[priority].clear()
                    else:
                        changes_to_save = []
                
                if changes_to_save:
                    await self._batch_save_changes(changes_to_save, priority)
                    
            except Exception as e:
                print(f"❌ Error en autosave loop {priority.name}: {e}")
                await asyncio.sleep(interval)  # Continuar después del error
    
    async def _batch_save_changes(self, changes: List[MemoryChange], priority: SavePriority):
        """Guarda cambios en lotes para eficiencia"""
        
        save_start_time = time.time()
        
        try:
            # Procesar en lotes del tamaño configurado
            for i in range(0, len(changes), self.config.batch_size):
                batch = changes[i:i + self.config.batch_size]
                
                # Ejecutar hasta config.concurrent_saves operaciones en paralelo
                tasks = []
                for j in range(0, len(batch), self.config.concurrent_saves):
                    concurrent_batch = batch[j:j + self.config.concurrent_saves]
                    task = asyncio.create_task(self._save_batch_to_db(concurrent_batch))
                    tasks.append(task)
                
                # Esperar a que todas las operaciones concurrentes terminen
                await asyncio.gather(*tasks)
            
            # Actualizar estadísticas
            save_time = time.time() - save_start_time
            self.save_stats["total_saves"] += len(changes)
            self.save_stats["average_save_time"] = (
                (self.save_stats["average_save_time"] * (self.save_stats["total_saves"] - len(changes)) + save_time) 
                / self.save_stats["total_saves"]
            )
            self.save_stats["last_save_timestamp"] = datetime.now()
            
            print(f"{priority.name}: {len(changes)} changes saved ({save_time:.3f}s)")
            
        except Exception as e:
            print(f"❌ Error guardando batch {priority.name}: {e}")
            self.save_stats["failed_saves"] += len(changes)
            
            # Reintentar cambios críticos
            if priority == SavePriority.CRITICAL:
                await self._retry_critical_changes(changes)
    
    async def _save_batch_to_db(self, batch: List[MemoryChange]):
        """Guarda un lote de cambios a la base de datos"""
        
        with self.db_lock:
            with sqlite3.connect(self.main_db_path) as conn:
                for change in batch:
                    try:
                        # Comprimir datos si está habilitado
                        if self.config.compression_enabled:
                            value_data = self._compress_data(change.data)
                        else:
                            value_data = pickle.dumps(change.data)
                        
                        # Insertar o actualizar memoria
                        conn.execute("""
                            INSERT OR REPLACE INTO agent_memory 
                            (agent_id, memory_type, key, value, checksum, timestamp, version)
                            VALUES (?, ?, ?, ?, ?, ?, 
                                COALESCE((SELECT version FROM agent_memory 
                                         WHERE agent_id=? AND memory_type=? AND key=?), 0) + 1)
                        """, (
                            change.agent_id,
                            change.memory_type.value,
                            change.change_id,  # Usar change_id como key
                            value_data,
                            change.checksum,
                            change.timestamp.timestamp(),
                            change.agent_id,
                            change.memory_type.value,
                            change.change_id
                        ))
                        
                        # Mark as saved in log
                        with sqlite3.connect(self.changes_db_path) as changes_conn:
                            changes_conn.execute("""
                                UPDATE memory_changes_log 
                                SET saved = TRUE 
                                WHERE change_id = ?
                            """, (change.change_id,))
                        
                    except Exception as e:
                        print(f"❌ Error guardando cambio {change.change_id}: {e}")
                        
                        # Incrementar contador de reintentos
                        with sqlite3.connect(self.changes_db_path) as changes_conn:
                            changes_conn.execute("""
                                UPDATE memory_changes_log 
                                SET retry_count = retry_count + 1 
                                WHERE change_id = ?
                            """, (change.change_id,))
    
    async def _retry_critical_changes(self, failed_changes: List[MemoryChange]):
        """Reintenta guardar cambios críticos que fallaron"""
        
        print(f"🔄 Reintentando {len(failed_changes)} cambios críticos...")
        
        for change in failed_changes:
            for attempt in range(self.config.retry_attempts):
                try:
                    await self._save_batch_to_db([change])
                    print(f"   {change.change_id} saved in attempt {attempt + 1}")
                    break
                except Exception as e:
                    print(f"   ❌ Intento {attempt + 1} falló para {change.change_id}: {e}")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))  # Backoff exponencial
    
    async def _force_save_buffer(self, priority: SavePriority):
        """Forces immediate save of a full buffer"""
        
        print(f"Forcing save of buffer {priority.name} (full)")
        
        with self.buffer_locks[priority]:
            if self.change_buffers[priority]:
                changes_to_save = self.change_buffers[priority].copy()
                self.change_buffers[priority].clear()
            else:
                changes_to_save = []
        
        if changes_to_save:
            await self._batch_save_changes(changes_to_save, priority)
    
    async def _incremental_backup_loop(self):
        """Loop de backups incrementales automáticos"""
        
        backup_interval = 3600  # 1 hora
        
        while True:
            try:
                await asyncio.sleep(backup_interval)
                await self._create_incremental_backup()
                
            except Exception as e:
                print(f"❌ Error en backup incremental: {e}")
                await asyncio.sleep(backup_interval)
    
    async def _create_incremental_backup(self):
        """Crea backup incremental de cambios recientes"""
        
        backup_start = time.time()
        
        # Obtener cambios desde último backup
        last_backup_time = await self._get_last_backup_timestamp()
        
        with sqlite3.connect(self.changes_db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM memory_changes_log 
                WHERE timestamp > ? AND saved = TRUE
                ORDER BY timestamp
            """, (last_backup_time,))
            
            recent_changes = cursor.fetchall()
        
        if not recent_changes:
            return
        
        # Comprimir datos de backup
        backup_data = {
            "changes": recent_changes,
            "timestamp": time.time(),
            "version": "1.0"
        }
        
        compressed_backup = self._compress_data(backup_data)
        
        # Guardar backup
        backup_id = f"backup_{int(time.time())}"
        metadata = {
            "backup_id": backup_id,
            "changes_count": len(recent_changes),
            "size_compressed": len(compressed_backup),
            "creation_time": time.time() - backup_start
        }
        
        with sqlite3.connect(self.backup_db_path) as conn:
            conn.execute("""
                INSERT INTO incremental_backups 
                (backup_id, timestamp, changes_count, compressed_data, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                backup_id,
                time.time(),
                len(recent_changes),
                compressed_backup,
                json.dumps(metadata)
            ))
        
        print(f"💾 Backup incremental creado: {len(recent_changes)} cambios ({len(compressed_backup)} bytes)")
    
    async def _get_last_backup_timestamp(self) -> float:
        """Obtiene timestamp del último backup"""
        
        try:
            with sqlite3.connect(self.backup_db_path) as conn:
                cursor = conn.execute("""
                    SELECT MAX(timestamp) FROM incremental_backups
                """)
                result = cursor.fetchone()
                return result[0] if result[0] else 0.0
                
        except Exception:
            return 0.0
    
    async def _maintenance_loop(self):
        """Loop de mantenimiento y optimización"""
        
        maintenance_interval = 86400  # 24 horas
        
        while True:
            try:
                await asyncio.sleep(maintenance_interval)
                await self._perform_maintenance()
                
            except Exception as e:
                print(f"❌ Error en mantenimiento: {e}")
                await asyncio.sleep(maintenance_interval)
    
    async def _perform_maintenance(self):
        """Realiza tareas de mantenimiento automático"""
        
        print("🧹 Iniciando mantenimiento automático...")
        
        # Limpiar logs antiguos
        cutoff_time = time.time() - (self.config.backup_retention_days * 86400)
        
        with sqlite3.connect(self.changes_db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM memory_changes_log 
                WHERE timestamp < ? AND saved = TRUE
            """, (cutoff_time,))
            deleted_logs = cursor.rowcount
        
        # Limpiar backups antiguos
        with sqlite3.connect(self.backup_db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM incremental_backups 
                WHERE timestamp < ?
            """, (cutoff_time,))
            deleted_backups = cursor.rowcount
        
        # Optimizar bases de datos
        for db_path in [self.main_db_path, self.changes_db_path, self.backup_db_path]:
            with sqlite3.connect(db_path) as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
        
        print(f"Maintenance completed:")
        print(f"   🗑️ Logs eliminados: {deleted_logs}")
        print(f"   🗑️ Backups eliminados: {deleted_backups}")
        print(f"   🔧 Bases de datos optimizadas")
    
    async def get_agent_memory(self, agent_id: str, memory_type: Optional[MemoryType] = None) -> Dict[str, Any]:
        """Recupera memoria actual de un agente"""
        
        with sqlite3.connect(self.main_db_path) as conn:
            if memory_type:
                cursor = conn.execute("""
                    SELECT key, value, timestamp, version 
                    FROM agent_memory 
                    WHERE agent_id = ? AND memory_type = ?
                    ORDER BY timestamp DESC
                """, (agent_id, memory_type.value))
            else:
                cursor = conn.execute("""
                    SELECT memory_type, key, value, timestamp, version 
                    FROM agent_memory 
                    WHERE agent_id = ?
                    ORDER BY memory_type, timestamp DESC
                """, (agent_id,))
            
            results = cursor.fetchall()
        
        # Reconstruir estructura de memoria
        agent_memory = {}
        
        for row in results:
            if memory_type:
                key, value_data, timestamp, version = row
                mem_type = memory_type.value
            else:
                mem_type, key, value_data, timestamp, version = row
            
            # Descomprimir datos
            try:
                if self.config.compression_enabled:
                    memory_data = self._decompress_data(value_data)
                else:
                    memory_data = pickle.loads(value_data)
                    
                if mem_type not in agent_memory:
                    agent_memory[mem_type] = {}
                    
                agent_memory[mem_type][key] = {
                    "data": memory_data,
                    "timestamp": datetime.fromtimestamp(timestamp),
                    "version": version
                }
                
            except Exception as e:
                print(f"⚠️ Error deserializando memoria {agent_id}/{mem_type}/{key}: {e}")
        
        return agent_memory
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Gets autosave system statistics"""
        
        # Contar cambios pendientes
        pending_changes = sum(len(buffer) for buffer in self.change_buffers.values())
        
        # Estadísticas de BD
        with sqlite3.connect(self.main_db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM agent_memory")
            total_memory_records = cursor.fetchone()[0]
        
        with sqlite3.connect(self.changes_db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memory_changes_log WHERE saved = FALSE")
            unsaved_changes = cursor.fetchone()[0]
        
        with sqlite3.connect(self.backup_db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM incremental_backups")
            total_backups = cursor.fetchone()[0]
        
        return {
            "system_status": "active",
            "pending_changes": pending_changes,
            "total_memory_records": total_memory_records,
            "unsaved_changes": unsaved_changes,
            "total_backups": total_backups,
            "save_stats": self.save_stats.copy(),
            "buffer_sizes": {
                priority.name: len(self.change_buffers[priority]) 
                for priority in SavePriority
            }
        }
    
    def register_save_callback(self, agent_id: str, memory_type: MemoryType, callback: Callable):
        """Registers callback for save events"""
        key = f"{agent_id}_{memory_type.value}"
        self.save_callbacks[key] = callback
        print(f"📧 Callback registrado para {key}")
    
    async def force_save_all(self):
        """Forces immediate save of all buffers"""
        
        print("Forcing save of all buffers...")
        
        tasks = []
        for priority in SavePriority:
            task = asyncio.create_task(self._force_save_buffer(priority))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        print("Forced save completed")
    
    async def shutdown(self):
        """Apaga el sistema de forma segura"""
        
        print("Shutting down autosave system...")
        
        # Force final save
        await self.force_save_all()
        
        # Cancelar tasks activos
        for task in self.save_tasks_active.values():
            task.cancel()
        
        # Crear backup final
        await self._create_incremental_backup()
        
        print("AutoSave system safely shut down")

# Integración con el sistema ágil
class AgileMemoryIntegration:
    """AutoSave integration with agile system"""
    
    def __init__(self, agile_engine, memory_orchestrator: MemoryAutoSaveOrchestrator):
        self.agile_engine = agile_engine
        self.memory_orchestrator = memory_orchestrator
        
        # Registrar callbacks para eventos ágiles
        self._setup_agile_callbacks()
    
    def _setup_agile_callbacks(self):
        """Configura callbacks para eventos del sistema ágil"""
        
        # Callback para cambios de tareas (alta prioridad)
        async def task_change_callback(change: MemoryChange):
            if "task_status" in change.data:
                print(f"Task change saved: {change.agent_id}")
        
        # Callback para daily standups (crítico)
        async def standup_callback(change: MemoryChange):
            if "standup" in change.data:
                print(f"Standup saved: {change.agent_id}")
        
        # Registrar callbacks
        self.memory_orchestrator.register_save_callback(
            "system", MemoryType.COLLABORATIVE, standup_callback
        )
    
    async def auto_save_sprint_progress(self, sprint_id: str):
        """Autoguarda progreso de sprint automáticamente"""
        
        sprint = self.agile_engine.sprints.get(sprint_id)
        if not sprint:
            return
        
        # Guardar estado de cada agente en el sprint
        for agent_id in sprint.team:
            agent_tasks = [task for task in sprint.tasks if task.assigned_agent == agent_id]
            
            # Preparar datos de memoria
            memory_data = {
                "sprint_id": sprint_id,
                "current_tasks": [asdict(task) for task in agent_tasks],
                "sprint_progress": len([t for t in agent_tasks if t.status.value == "done"]) / len(agent_tasks) if agent_tasks else 0,
                "last_standup": sprint.daily_standups[-1] if sprint.daily_standups else None
            }
            
            # Registrar cambio con prioridad alta
            await self.memory_orchestrator.register_memory_change(
                agent_id=agent_id,
                memory_type=MemoryType.WORKING,
                data=memory_data,
                priority=SavePriority.HIGH
            )
    
    async def save_critical_sprint_events(self, sprint_id: str, event_type: str, event_data: Dict[str, Any]):
        """Guarda eventos críticos del sprint inmediatamente"""
        
        await self.memory_orchestrator.register_memory_change(
            agent_id="system",
            memory_type=MemoryType.EPISODIC,
            data={
                "sprint_id": sprint_id,
                "event_type": event_type,
                "event_data": event_data,
                "timestamp": datetime.now().isoformat()
            },
            priority=SavePriority.CRITICAL
        )

# Complete system demo
async def demo_memory_autosave():
    """Memory autosave system demo"""
    
    print("DEMO: Memory Autosave System")
    print("="*60)
    
    # Initialize system
    config = AutoSaveConfig(
        critical_save_interval=0.5,  # Faster for demo
        high_save_interval=2.0,
        medium_save_interval=5.0,
        low_save_interval=10.0
    )
    
    memory_system = MemoryAutoSaveOrchestrator(config=config)
    
    # Simulate memory changes from multiple agents
    agents = ["agent_01", "agent_02", "agent_03"]
    
    for i, agent_id in enumerate(agents):
        # Simulate different memory types
        await memory_system.register_memory_change(
            agent_id=agent_id,
            memory_type=MemoryType.WORKING,
            data={"current_task": f"task_{i}", "progress": 0.5},
            priority=SavePriority.HIGH
        )
        
        await memory_system.register_memory_change(
            agent_id=agent_id,
            memory_type=MemoryType.EPISODIC,
            data={"event": f"standup_completed", "timestamp": datetime.now().isoformat()},
            priority=SavePriority.CRITICAL
        )
    
    # Wait for some saves to be processed
    await asyncio.sleep(3)
    
    # Get statistics
    stats = await memory_system.get_system_stats()
    
    # Get agent memory
    agent_memory = await memory_system.get_agent_memory("agent_01")
    
    # Force save all
    await memory_system.force_save_all()
    
    print(f"\nSTATISTICS:")
    print(f"   Memory records: {stats['total_memory_records']}")
    print(f"   Pending changes: {stats['pending_changes']}")
    print(f"   Total saved: {stats['save_stats']['total_saves']}")
    print(f"   Average time: {stats['save_stats']['average_save_time']:.3f}s")
    print(f"   Backups: {stats['total_backups']}")
    
    await memory_system.shutdown()

if __name__ == "__main__":
    asyncio.run(demo_memory_autosave())