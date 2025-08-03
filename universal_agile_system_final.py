#!/usr/bin/env python3
"""
Universal Agile System Final - Complete Universal Agile System Production-Ready
Integrates ALL components: automatic detection, deep agile methodology,
health analysis, intelligent filters and multi-project management
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import all developed components
from universal_project_detector import UniversalProjectDetector, DetectedProject
from deep_agile_methodology import DeepAgileOrchestrator, UserStory, Epic
from memory_autosave_system import MemoryAutoSaveOrchestrator, AutoSaveConfig
from context_liberation_system import ContextLiberationOrchestrator
from quick_project_health_check import QuickHealthAnalyzer

class UniversalAgileSystemFinal:
    """Final Universal Agile System - Production Ready"""
    
    def __init__(self, storage_path: str = "./universal_agile_system"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Main components
        self.project_detector = UniversalProjectDetector()
        self.health_analyzer = QuickHealthAnalyzer()
        
        # System state
        self.projects: Dict[str, Dict[str, Any]] = {}  # project_id -> project_data
        self.active_projects: List[str] = []
        
        # Global metrics
        self.global_metrics = {
            "total_projects": 0,
            "total_agents": 0,
            "total_user_stories": 0,
            "total_epics": 0,
            "average_health_score": 0.0,
            "active_sprints": 0,
            "system_started": datetime.now().isoformat()
        }
    
    async def discover_and_onboard_project(self, project_path: str, 
                                         custom_config: Optional[Dict[str, Any]] = None) -> str:
        """Discovers, analyzes and configures a project completely"""
        
        print(f"🚀 ONBOARDING COMPLETO DE PROYECTO")
        print("="*70)
        print(f"📁 Path: {project_path}")
        
        # 1. Análisis de salud del proyecto
        print("\nSTEP 1: Health Analysis")
        health_result = self.health_analyzer.analyze_project_quick(project_path)
        
        if "error" in health_result:
            raise ValueError(f"Error en análisis de salud: {health_result['error']}")
        
        # 2. Detección de estructura
        print("\nSTEP 2: Structure Detection")
        detected_project = await self.project_detector.detect_project_structure(project_path)
        
        # 3. Configuración ágil automática
        print("\nSTEP 3: Agile Configuration")
        agile_config = self._generate_smart_agile_config(detected_project, health_result, custom_config)
        
        # 4. Inicialización de componentes
        print("\nSTEP 4: System Initialization")
        project_systems = await self._initialize_project_systems(detected_project, agile_config)
        
        # 5. Generación automática de backlog
        print("\nSTEP 5: Product Backlog Generation")
        initial_backlog = await self._generate_intelligent_backlog(
            detected_project, project_systems["agile_orchestrator"], health_result
        )
        
        # 6. Almacenar proyecto
        project_id = detected_project.id
        self.projects[project_id] = {
            "detected_project": detected_project,
            "health_analysis": health_result,
            "agile_config": agile_config,
            "systems": project_systems,
            "initial_backlog": initial_backlog,
            "onboarded_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        self.active_projects.append(project_id)
        
        # 7. Actualizar métricas globales
        self._update_global_metrics(detected_project, health_result, initial_backlog)
        
        # 8. Guardar configuración
        await self._save_project_complete_config(project_id)
        
        print(f"\nPROJECT ONBOARDED SUCCESSFULLY:")
        print(f"   📋 ID: {project_id}")
        print(f"   Name: {detected_project.name}")
        print(f"   💚 Health Score: {health_result['analysis_summary']['health_score']}%")
        print(f"   👥 Agentes: {detected_project.total_agents}")
        print(f"   📚 Epics: {len([epic for epic in project_systems['agile_orchestrator'].epics.values()])}")
        print(f"   📝 User Stories: {len(initial_backlog)}")
        
        return project_id
    
    def _generate_smart_agile_config(self, project: DetectedProject, 
                                   health_result: Dict[str, Any],
                                   custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera configuración ágil inteligente basada en detección y salud"""
        
        # Configuración base
        config = {
            "project_id": project.id,
            "project_name": project.name,
            "project_type": project.project_type.value,
            "structure_depth": project.structure_depth.value,
            "detection_confidence": project.detection_confidence
        }
        
        # Configuración de roles automática
        if project.coordinators:
            config["product_owner"] = project.coordinators[0].id
            config["scrum_master"] = project.coordinators[1].id if len(project.coordinators) > 1 else project.coordinators[0].id
        else:
            config["product_owner"] = "auto_product_owner"
            config["scrum_master"] = "auto_scrum_master"
        
        # Equipo inteligente (evitar equipos masivos)
        team_members = []
        
        # Añadir coordinadores
        team_members.extend([coord.id for coord in project.coordinators])
        
        # Añadir líderes de squads
        for squad in project.squads:
            team_members.extend(squad.leader_agents[:2])  # Máximo 2 líderes por squad
        
        # Añadir algunos agentes individuales
        team_members.extend([agent.id for agent in project.individual_agents[:5]])
        
        # Remover duplicados y limitar tamaño
        team_members = list(set(team_members))[:20]  # Máximo 20 miembros
        
        config["team_members"] = team_members
        config["team_size"] = len(team_members)
        
        # Configuración de sprints basada en salud y tamaño
        health_score = health_result["analysis_summary"]["health_score"]
        
        if health_score >= 90:
            # Proyecto saludable = sprints más largos y ambiciosos
            config["default_sprint_duration"] = 14
            config["sprint_capacity_per_member"] = 3.0
        elif health_score >= 70:
            # Proyecto OK = sprints estándar
            config["default_sprint_duration"] = 10
            config["sprint_capacity_per_member"] = 2.5
        else:
            # Proyecto con issues = sprints cortos y conservadores
            config["default_sprint_duration"] = 7
            config["sprint_capacity_per_member"] = 2.0
        
        # Configuración de ceremonies basada en tamaño del equipo
        if len(team_members) > 15:
            config["ceremonies"] = {
                "daily_standup_duration": 20,  # Más tiempo para equipos grandes
                "planning_duration_hours": 6,
                "review_duration_hours": 3,
                "retrospective_duration_hours": 2
            }
        else:
            config["ceremonies"] = {
                "daily_standup_duration": 15,
                "planning_duration_hours": 4,
                "review_duration_hours": 2,
                "retrospective_duration_hours": 1
            }
        
        # Configuración de persistencia basada en complejidad
        if project.total_agents > 50:
            config["autosave_config"] = {
                "critical_save_interval": 0.05,
                "high_save_interval": 2.0,
                "medium_save_interval": 15.0,
                "compression_enabled": True,
                "max_memory_buffer": 200
            }
        else:
            config["autosave_config"] = {
                "critical_save_interval": 0.1,
                "high_save_interval": 5.0,
                "medium_save_interval": 30.0,
                "compression_enabled": True,
                "max_memory_buffer": 100
            }
        
        # Aplicar configuración personalizada
        if custom_config:
            config.update(custom_config)
        
        print(f"⚙️ Configuración inteligente generada:")
        print(f"   👑 Product Owner: {config['product_owner']}")
        print(f"   Scrum Master: {config['scrum_master']}")
        print(f"   👥 Equipo: {config['team_size']} miembros")
        print(f"   📅 Sprints: {config['default_sprint_duration']} días")
        print(f"   💚 Basado en health: {health_score}%")
        
        return config
    
    async def _initialize_project_systems(self, project: DetectedProject, 
                                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Inicializa todos los sistemas del proyecto"""
        
        # Crear directorio del proyecto
        project_dir = self.storage_path / project.id
        project_dir.mkdir(exist_ok=True)
        
        # 1. Sistema ágil principal
        agile_orchestrator = DeepAgileOrchestrator(
            product_owner=config["product_owner"],
            scrum_master=config["scrum_master"],
            team=config["team_members"]
        )
        
        # 2. Sistema de memoria con configuración optimizada
        memory_config = AutoSaveConfig(**config["autosave_config"])
        memory_system = MemoryAutoSaveOrchestrator(
            storage_path=str(project_dir / "memory"),
            config=memory_config
        )
        
        # 3. Sistema de contexto
        context_system = ContextLiberationOrchestrator(
            storage_path=str(project_dir / "context")
        )
        
        systems = {
            "agile_orchestrator": agile_orchestrator,
            "memory_system": memory_system,
            "context_system": context_system
        }
        
        print(f"🛠️ Sistemas inicializados:")
        print(f"   Agile Orchestrator: Active")
        print(f"   Memory System: Active")
        print(f"   Context System: Active")
        
        return systems
    
    async def _generate_intelligent_backlog(self, project: DetectedProject,
                                          agile_orchestrator: DeepAgileOrchestrator,
                                          health_result: Dict[str, Any]) -> List[UserStory]:
        """Genera backlog inteligente basado en estructura y salud del proyecto"""
        
        # Epic principal
        epic_title = f"Sistema Operativo - {project.name}"
        epic_goal = f"Crear sistema operativo completo para {project.name} que maximice valor y eficiencia"
        
        main_epic = await agile_orchestrator.create_epic(
            title=epic_title,
            description=f"Epic principal para transformar {project.name} en sistema operativo ágil",
            goal=epic_goal,
            business_value=95
        )
        
        generated_stories = []
        
        # 1. Historias basadas en issues de salud (alta prioridad)
        health_stories = await self._generate_health_improvement_stories(
            health_result, main_epic.id, agile_orchestrator
        )
        generated_stories.extend(health_stories)
        
        # 2. Historias de configuración del sistema
        system_stories = await self._generate_system_setup_stories(
            project, main_epic.id, agile_orchestrator
        )
        generated_stories.extend(system_stories)
        
        # 3. Historias por squad (selección inteligente)
        squad_stories = await self._generate_intelligent_squad_stories(
            project, main_epic.id, agile_orchestrator
        )
        generated_stories.extend(squad_stories)
        
        # 4. Historias de integración y coordinación
        integration_stories = await self._generate_integration_stories(
            project, main_epic.id, agile_orchestrator
        )
        generated_stories.extend(integration_stories)
        
        # 5. Historias de valor de negocio
        business_stories = await self._generate_business_value_stories(
            project, main_epic.id, agile_orchestrator
        )
        generated_stories.extend(business_stories)
        
        print(f"📝 Backlog inteligente generado:")
        print(f"   📚 1 Epic principal")
        print(f"   🏥 {len(health_stories)} historias de salud")
        print(f"   ⚙️ {len(system_stories)} historias de sistema")
        print(f"   🏢 {len(squad_stories)} historias de squads")
        print(f"   🔗 {len(integration_stories)} historias de integración")
        print(f"   💼 {len(business_stories)} historias de negocio")
        print(f"   📝 {len(generated_stories)} total User Stories")
        
        return generated_stories
    
    async def _generate_health_improvement_stories(self, health_result: Dict[str, Any],
                                                 epic_id: str, orchestrator: DeepAgileOrchestrator) -> List[UserStory]:
        """Genera historias para mejorar salud del proyecto"""
        
        stories = []
        issues = health_result["issues_found"]
        
        # Historia para archivos deprecados
        if issues["deprecated_patterns"] > 0:
            story = await orchestrator.create_user_story(
                title="Limpieza de Archivos Deprecados",
                as_a="desarrollador del equipo",
                i_want=f"limpiar {issues['deprecated_patterns']} archivos con patrones deprecados",
                so_that="el proyecto tenga mejor mantenibilidad y claridad",
                epic_id=epic_id,
                business_value=80
            )
            
            await orchestrator.add_acceptance_criteria(
                story.id,
                "Archivos deprecados removidos",
                given="Archivos con patrones _old, _deprecated, _legacy existen",
                when="Se ejecuta la limpieza",
                then="Los archivos son archivados o removidos apropiadamente"
            )
            
            stories.append(story)
        
        # Historia para archivos antiguos
        if issues["very_old_files"] > 5:
            story = await orchestrator.create_user_story(
                title="Auditoría de Archivos Antiguos",
                as_a="administrador del proyecto",
                i_want=f"revisar {issues['very_old_files']} archivos muy antiguos",
                so_that="pueda determinar cuáles siguen siendo relevantes",
                epic_id=epic_id,
                business_value=60
            )
            stories.append(story)
        
        # Historia para archivos grandes
        if issues["large_files"] > 0:
            story = await orchestrator.create_user_story(
                title="Optimización de Archivos Grandes",
                as_a="desarrollador",
                i_want=f"optimizar {issues['large_files']} archivos grandes",
                so_that="el sistema sea más eficiente y fácil de mantener",
                epic_id=epic_id,
                business_value=70
            )
            stories.append(story)
        
        return stories
    
    async def _generate_system_setup_stories(self, project: DetectedProject,
                                           epic_id: str, orchestrator: DeepAgileOrchestrator) -> List[UserStory]:
        """Genera historias de configuración del sistema"""
        
        stories = []
        
        # Historia principal de configuración
        story = await orchestrator.create_user_story(
            title="Configuración del Sistema Ágil",
            as_a="scrum master",
            i_want="configurar el sistema ágil para el proyecto",
            so_that="el equipo pueda trabajar eficientemente con metodología ágil",
            epic_id=epic_id,
            business_value=90
        )
        
        await orchestrator.add_acceptance_criteria(
            story.id,
            "Sistema ágil configurado",
            given="El proyecto está detectado y analizado",
            when="Se configura el sistema ágil",
            then="Todos los componentes están operativos y sincronizados"
        )
        
        stories.append(story)
        
        # Historia de monitoreo
        story = await orchestrator.create_user_story(
            title="Sistema de Monitoreo y Métricas",
            as_a="product owner",
            i_want="monitorear el progreso y métricas del equipo",
            so_that="pueda tomar decisiones informadas sobre el producto",
            epic_id=epic_id,
            business_value=75
        )
        stories.append(story)
        
        return stories
    
    async def _generate_intelligent_squad_stories(self, project: DetectedProject,
                                                epic_id: str, orchestrator: DeepAgileOrchestrator) -> List[UserStory]:
        """Genera historias inteligentes para squads (solo los más importantes)"""
        
        stories = []
        
        # Seleccionar top 5 squads por importancia
        sorted_squads = sorted(
            project.squads, 
            key=lambda s: len(s.agents) + len(s.leader_agents),
            reverse=True
        )
        
        for squad in sorted_squads[:5]:  # Solo top 5
            if len(squad.agents) > 0:  # Solo squads con agentes
                story = await orchestrator.create_user_story(
                    title=f"Operatividad del Squad {squad.name}",
                    as_a="líder del squad",
                    i_want=f"que el squad {squad.name} opere eficientemente",
                    so_that=f"pueda entregar valor en {squad.purpose}",
                    epic_id=epic_id,
                    business_value=70 - len(stories) * 5  # Prioridad decreciente
                )
                stories.append(story)
        
        return stories
    
    async def _generate_integration_stories(self, project: DetectedProject,
                                          epic_id: str, orchestrator: DeepAgileOrchestrator) -> List[UserStory]:
        """Genera historias de integración entre componentes"""
        
        stories = []
        
        # Historia de integración entre squads
        if len(project.squads) > 1:
            story = await orchestrator.create_user_story(
                title="Integración Inter-Squad",
                as_a="arquitecto del sistema",
                i_want=f"que los {len(project.squads)} squads trabajen coordinadamente",
                so_that="el sistema opere como una unidad cohesiva",
                epic_id=epic_id,
                business_value=85
            )
            stories.append(story)
        
        # Historia de coordinación con coordinadores
        if project.coordinators:
            story = await orchestrator.create_user_story(
                title="Sistema de Coordinación Global",
                as_a="coordinador del sistema",
                i_want="coordinar todas las actividades del proyecto",
                so_that="se maximice la eficiencia y se eviten conflictos",
                epic_id=epic_id,
                business_value=80
            )
            stories.append(story)
        
        return stories
    
    async def _generate_business_value_stories(self, project: DetectedProject,
                                             epic_id: str, orchestrator: DeepAgileOrchestrator) -> List[UserStory]:
        """Genera historias enfocadas en valor de negocio"""
        
        stories = []
        
        # Historia de métricas de valor
        story = await orchestrator.create_user_story(
            title="Métricas de Valor de Negocio",
            as_a="stakeholder",
            i_want="ver métricas claras del valor generado por el proyecto",
            so_that="pueda evaluar el ROI y tomar decisiones estratégicas",
            epic_id=epic_id,
            business_value=85
        )
        stories.append(story)
        
        # Historia de optimización de procesos
        story = await orchestrator.create_user_story(
            title="Optimización de Procesos",
            as_a="usuario del sistema",
            i_want="que los procesos del proyecto sean lo más eficientes posible",
            so_that="se reduzcan tiempos y costos operativos",
            epic_id=epic_id,
            business_value=75
        )
        stories.append(story)
        
        return stories
    
    def _update_global_metrics(self, project: DetectedProject, 
                             health_result: Dict[str, Any], 
                             backlog: List[UserStory]):
        """Actualiza métricas globales del sistema"""
        
        self.global_metrics["total_projects"] += 1
        self.global_metrics["total_agents"] += project.total_agents
        self.global_metrics["total_user_stories"] += len(backlog)
        self.global_metrics["total_epics"] += 1  # Epic principal
        
        # Calcular promedio de health score
        current_avg = self.global_metrics["average_health_score"]
        total_projects = self.global_metrics["total_projects"]
        new_score = health_result["analysis_summary"]["health_score"]
        
        self.global_metrics["average_health_score"] = (
            (current_avg * (total_projects - 1) + new_score) / total_projects
        )
    
    async def _save_project_complete_config(self, project_id: str):
        """Guarda configuración completa del proyecto"""
        
        config_path = self.storage_path / project_id / "complete_config.json"
        config_path.parent.mkdir(exist_ok=True)
        
        project_data = self.projects[project_id]
        
        # Preparar datos serializables
        config_data = {
            "project_id": project_id,
            "project_name": project_data["detected_project"].name,
            "agile_config": project_data["agile_config"],
            "health_summary": project_data["health_analysis"]["analysis_summary"],
            "backlog_size": len(project_data["initial_backlog"]),
            "onboarded_at": project_data["onboarded_at"],
            "system_version": "1.0.0"
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"💾 Configuración completa guardada: {config_path}")
    
    async def create_sprint_for_project(self, project_id: str, sprint_name: str) -> Dict[str, Any]:
        """Crea sprint para un proyecto específico"""
        
        if project_id not in self.projects:
            raise ValueError(f"Proyecto {project_id} no encontrado")
        
        project_data = self.projects[project_id]
        orchestrator = project_data["systems"]["agile_orchestrator"]
        config = project_data["agile_config"]
        
        print(f"\n📋 CREANDO SPRINT: {sprint_name}")
        print(f"📁 Proyecto: {project_data['detected_project'].name}")
        
        # Realizar Sprint Planning
        sprint_plan = await orchestrator.plan_sprint(config["default_sprint_duration"])
        
        # Actualizar métricas
        self.global_metrics["active_sprints"] += 1
        
        sprint_result = {
            "project_id": project_id,
            "sprint_name": sprint_name,
            "plan": sprint_plan,
            "config": config,
            "created_at": datetime.now().isoformat()
        }
        
        print(f"Sprint created successfully")
        
        return sprint_result
    
    async def get_system_dashboard(self) -> Dict[str, Any]:
        """Obtiene dashboard completo del sistema universal"""
        
        # Métricas por proyecto
        projects_summary = {}
        for project_id in self.active_projects:
            project_data = self.projects[project_id]
            projects_summary[project_id] = {
                "name": project_data["detected_project"].name,
                "health_score": project_data["health_analysis"]["analysis_summary"]["health_score"],
                "team_size": project_data["agile_config"]["team_size"],
                "agents_detected": project_data["detected_project"].total_agents,
                "backlog_size": len(project_data["initial_backlog"]),
                "status": project_data["status"]
            }
        
        dashboard = {
            "system_info": {
                "version": "1.0.0 - Production Ready",
                "started_at": self.global_metrics["system_started"],
                "uptime_hours": (datetime.now() - datetime.fromisoformat(self.global_metrics["system_started"])).total_seconds() / 3600
            },
            "global_metrics": self.global_metrics,
            "projects": projects_summary,
            "system_health": {
                "total_active_projects": len(self.active_projects),
                "average_project_health": self.global_metrics["average_health_score"],
                "total_capacity": sum(p["team_size"] for p in projects_summary.values()),
                "system_status": "operational"
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return dashboard

# Función principal de demo
async def demo_universal_agile_final():
    """Demo del sistema ágil universal final"""
    
    print("🌟 SISTEMA ÁGIL UNIVERSAL FINAL - PRODUCTION READY")
    print("="*80)
    
    # Crear sistema universal
    system = UniversalAgileSystemFinal()
    
    # Onboarding completo de proyecto
    project_path = "/mnt/c/Users/pedro/Desktop/agente/Diamond-Team-Clean"
    
    project_id = await system.discover_and_onboard_project(
        project_path=project_path,
        custom_config={
            "product_owner": "Pedro_Product_Owner",
            "scrum_master": "Ana_Scrum_Master",
            "default_sprint_duration": 14
        }
    )
    
    # Crear sprint
    sprint_result = await system.create_sprint_for_project(
        project_id=project_id,
        sprint_name="Launch Sprint - Sistema Operativo v1.0"
    )
    
    # Dashboard final
    dashboard = await system.get_system_dashboard()
    
    print(f"\nOPERATING SYSTEM DEPLOYED:")
    print(f"   🚀 Proyectos activos: {dashboard['system_health']['total_active_projects']}")
    print(f"   💚 Health promedio: {dashboard['global_metrics']['average_health_score']:.1f}%")
    print(f"   👥 Capacidad total: {dashboard['system_health']['total_capacity']} miembros")
    print(f"   📚 Total epics: {dashboard['global_metrics']['total_epics']}")
    print(f"   📝 Total user stories: {dashboard['global_metrics']['total_user_stories']}")
    print(f"   ⚡ Sprints activos: {dashboard['global_metrics']['active_sprints']}")
    
    return system, dashboard

if __name__ == "__main__":
    asyncio.run(demo_universal_agile_final())