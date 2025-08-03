#!/usr/bin/env python3
"""
Universal Project Detector - Automatic Project Structure Detection
Detects and automatically adapts any team/project structure (2-3 levels)
"""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import re

class ProjectType(Enum):
    AGENT_TEAM = "agent_team"           # AI agent teams
    SOFTWARE_PROJECT = "software_project"  # Traditional software projects
    RESEARCH_GROUP = "research_group"   # Research groups
    BUSINESS_UNITS = "business_units"   # Business units
    MIXED_TEAM = "mixed_team"          # Mixed teams

class StructureDepth(Enum):
    TWO_LEVEL = 2    # Coordinator > Agents
    THREE_LEVEL = 3  # Coordinator > Squad Leaders > Agents

@dataclass
class DetectedAgent:
    """Automatically detected agent/member"""
    id: str
    name: str
    path: Path
    role: str = "agent"
    capabilities: List[str] = field(default_factory=list)
    config_file: Optional[Path] = None
    launch_script: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectedSquad:
    """Automatically detected squad/team"""
    id: str
    name: str
    path: Path
    purpose: str = ""
    agents: List[DetectedAgent] = field(default_factory=list)
    leader_agents: List[str] = field(default_factory=list)
    config_file: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectedProject:
    """Automatically detected project"""
    id: str
    name: str
    root_path: Path
    project_type: ProjectType
    structure_depth: StructureDepth
    
    # Detected hierarchical structure
    coordinators: List[DetectedAgent] = field(default_factory=list)
    squads: List[DetectedSquad] = field(default_factory=list)
    individual_agents: List[DetectedAgent] = field(default_factory=list)
    
    # Detected configuration
    config_files: List[Path] = field(default_factory=list)
    total_agents: int = 0
    detection_confidence: float = 0.0
    
    # Additional metadata
    technologies: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)

class UniversalProjectDetector:
    """Universal project structure detector"""
    
    def __init__(self):
        self.detection_patterns = self._init_detection_patterns()
        self.file_indicators = self._init_file_indicators()
        self.structure_heuristics = self._init_structure_heuristics()
    
    def _init_detection_patterns(self) -> Dict[str, Any]:
        """Patterns for detecting different project types"""
        return {
            "agent_indicators": [
                "agent", "bot", "ai", "assistant", "specialist", "coordinator",
                "orchestrator", "manager", "handler", "processor"
            ],
            "config_files": [
                "config.json", "config.yaml", "config.yml", "settings.json",
                "agent_config.json", "team_config.yaml", ".env", "Dockerfile"
            ],
            "launch_scripts": [
                "launch.sh", "start.sh", "run.sh", "launch_agent.sh",
                "main.py", "app.py", "agent.py", "run.py"
            ],
            "structure_keywords": {
                "coordinators": ["coordinator", "orchestrator", "manager", "lead", "chief"],
                "leaders": ["leader", "head", "senior", "principal", "supervisor"],
                "squads": ["squad", "team", "group", "unit", "department", "division"],
                "agents": ["agent", "specialist", "worker", "member", "contributor"]
            }
        }
    
    def _init_file_indicators(self) -> Dict[str, List[str]]:
        """File indicators for different technologies"""
        return {
            "python": [".py", "requirements.txt", "pyproject.toml", "setup.py"],
            "nodejs": ["package.json", "package-lock.json", ".js", ".ts"],
            "docker": ["Dockerfile", "docker-compose.yml", ".dockerignore"],
            "ai_ml": ["model.pkl", ".ipynb", "training.py", "inference.py"],
            "config": [".json", ".yaml", ".yml", ".toml", ".ini", ".env"]
        }
    
    def _init_structure_heuristics(self) -> Dict[str, Any]:
        """Heuristics for determining organizational structure"""
        return {
            "depth_indicators": {
                2: ["direct_agents", "flat_structure", "small_team"],
                3: ["squad_based", "hierarchical", "large_team", "departments"]
            },
            "size_thresholds": {
                "small": 10,      # ≤10 agents = probably 2 levels
                "medium": 50,     # 10-50 agents = probably 3 levels
                "large": 100      # >50 agents = definitely 3 levels
            }
        }
    
    async def detect_project_structure(self, root_path: str) -> DetectedProject:
        """Automatically detects the complete project structure"""
        
        root = Path(root_path).resolve()
        if not root.exists():
            raise ValueError(f"Path does not exist: {root_path}")
        
        print(f"DETECTING PROJECT STRUCTURE: {root.name}")
        print("="*60)
        
        # 1. Initial directory analysis
        initial_analysis = await self._analyze_root_directory(root)
        
        # 2. Detect project type
        project_type = self._determine_project_type(initial_analysis)
        print(f"Detected type: {project_type.value}")
        
        # 3. Detect hierarchical structure
        structure_data = await self._detect_hierarchical_structure(root, project_type)
        
        # 4. Determine depth
        structure_depth = self._determine_structure_depth(structure_data)
        print(f"Depth: {structure_depth.value} levels")
        
        # 5. Detect agents/members
        agents_data = await self._detect_agents_and_squads(root, structure_data)
        
        # 6. Detect configurations
        config_data = await self._detect_configurations(root)
        
        # 7. Calculate detection confidence
        confidence = self._calculate_detection_confidence(initial_analysis, structure_data, agents_data)
        
        # 8. Create detected project
        detected_project = DetectedProject(
            id=self._generate_project_id(root),
            name=root.name,
            root_path=root,
            project_type=project_type,
            structure_depth=structure_depth,
            coordinators=agents_data["coordinators"],
            squads=agents_data["squads"],
            individual_agents=agents_data["individual_agents"],
            config_files=config_data["config_files"],
            total_agents=agents_data["total_count"],
            detection_confidence=confidence,
            technologies=config_data["technologies"],
            languages=config_data["languages"],
            frameworks=config_data["frameworks"]
        )
        
        print(f"Detection completed:")
        print(f"   {detected_project.total_agents} agents/members")
        print(f"   {len(detected_project.squads)} squads/teams")
        print(f"   {len(detected_project.coordinators)} coordinators")
        print(f"   Confidence: {confidence*100:.1f}%")
        
        return detected_project
    
    async def _analyze_root_directory(self, root: Path) -> Dict[str, Any]:
        """Initial root directory analysis"""
        
        analysis = {
            "total_subdirs": 0,
            "total_files": 0,
            "subdirectories": [],
            "file_types": {},
            "naming_patterns": [],
            "potential_agents": [],
            "potential_squads": []
        }
        
        try:
            # Scan directory
            for item in root.iterdir():
                if item.is_dir():
                    analysis["total_subdirs"] += 1
                    analysis["subdirectories"].append(item)
                    
                    # Detect naming patterns
                    dir_name = item.name.lower()
                    if any(keyword in dir_name for keyword in self.detection_patterns["agent_indicators"]):
                        analysis["potential_agents"].append(item)
                    
                    if any(keyword in dir_name for keyword in self.detection_patterns["structure_keywords"]["squads"]):
                        analysis["potential_squads"].append(item)
                
                elif item.is_file():
                    analysis["total_files"] += 1
                    file_ext = item.suffix.lower()
                    analysis["file_types"][file_ext] = analysis["file_types"].get(file_ext, 0) + 1
        
        except PermissionError:
            print(f"Warning: No permissions to access {root}")
        
        return analysis
    
    def _determine_project_type(self, analysis: Dict[str, Any]) -> ProjectType:
        """Determines project type based on analysis"""
        
        # Count indicators of different types
        agent_indicators = len(analysis["potential_agents"])
        squad_indicators = len(analysis["potential_squads"])
        
        # Check typical configuration files
        has_ai_configs = any(".json" in ext or ".yaml" in ext for ext in analysis["file_types"])
        has_python = ".py" in analysis["file_types"]
        has_docker = any("dockerfile" in str(subdir).lower() for subdir in analysis["subdirectories"])
        
        # Decision logic
        if agent_indicators >= 3 and (has_ai_configs or has_python):
            return ProjectType.AGENT_TEAM
        elif squad_indicators >= 2 and agent_indicators >= 1:
            return ProjectType.MIXED_TEAM
        elif has_python and has_ai_configs:
            return ProjectType.SOFTWARE_PROJECT
        elif analysis["total_subdirs"] >= 5:
            return ProjectType.BUSINESS_UNITS
        else:
            return ProjectType.MIXED_TEAM  # Default fallback
    
    async def _detect_hierarchical_structure(self, root: Path, project_type: ProjectType) -> Dict[str, Any]:
        """Detects the hierarchical structure of the project"""
        
        structure = {
            "levels": [],
            "hierarchy_map": {},
            "organization_pattern": "flat"
        }
        
        # Level 1: Root directory
        level_1 = {"path": root, "type": "root", "children": []}
        
        # Level 2: Main subdirectories
        level_2_items = []
        for subdir in root.iterdir():
            if subdir.is_dir():
                item_data = await self._analyze_directory_depth(subdir, depth=2)
                level_2_items.append(item_data)
                level_1["children"].append(item_data)
        
        structure["levels"].append(level_1)
        
        # Determine if level 3 exists
        has_level_3 = False
        for item in level_2_items:
            if item["subdirectory_count"] >= 3:  # If it has enough subdirectories
                has_level_3 = True
                # Analyze level 3
                level_3_items = []
                for subdir in item["path"].iterdir():
                    if subdir.is_dir():
                        item_3 = await self._analyze_directory_depth(subdir, depth=3)
                        level_3_items.append(item_3)
                item["children"] = level_3_items
        
        if has_level_3:
            structure["organization_pattern"] = "hierarchical"
        else:
            structure["organization_pattern"] = "flat"
        
        return structure
    
    async def _analyze_directory_depth(self, directory: Path, depth: int) -> Dict[str, Any]:
        """Analyzes a directory at a certain depth"""
        
        analysis = {
            "path": directory,
            "name": directory.name,
            "depth": depth,
            "subdirectory_count": 0,
            "file_count": 0,
            "config_files": [],
            "launch_scripts": [],
            "potential_role": "unknown",
            "children": []
        }
        
        try:
            items = list(directory.iterdir())
            
            for item in items:
                if item.is_dir():
                    analysis["subdirectory_count"] += 1
                elif item.is_file():
                    analysis["file_count"] += 1
                    
                    # Detect important files
                    filename = item.name.lower()
                    if any(config in filename for config in self.detection_patterns["config_files"]):
                        analysis["config_files"].append(item)
                    
                    if any(script in filename for script in self.detection_patterns["launch_scripts"]):
                        analysis["launch_scripts"].append(item)
            
            # Determine potential role based on name
            dir_name = directory.name.lower()
            if any(coord in dir_name for coord in self.detection_patterns["structure_keywords"]["coordinators"]):
                analysis["potential_role"] = "coordinator"
            elif any(leader in dir_name for leader in self.detection_patterns["structure_keywords"]["leaders"]):
                analysis["potential_role"] = "leader"
            elif any(squad in dir_name for squad in self.detection_patterns["structure_keywords"]["squads"]):
                analysis["potential_role"] = "squad"
            elif any(agent in dir_name for agent in self.detection_patterns["structure_keywords"]["agents"]):
                analysis["potential_role"] = "agent"
        
        except PermissionError:
            pass
        
        return analysis
    
    def _determine_structure_depth(self, structure_data: Dict[str, Any]) -> StructureDepth:
        """Determines the depth of the organizational structure"""
        
        if structure_data["organization_pattern"] == "hierarchical":
            return StructureDepth.THREE_LEVEL
        else:
            return StructureDepth.TWO_LEVEL
    
    async def _detect_agents_and_squads(self, root: Path, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detects agents and squads based on structure"""
        
        agents_data = {
            "coordinators": [],
            "squads": [],
            "individual_agents": [],
            "total_count": 0
        }
        
        # Traverse detected structure
        if structure_data["organization_pattern"] == "hierarchical":
            # Estructura de 3 niveles
            await self._detect_three_level_structure(root, agents_data)
        else:
            # Estructura de 2 niveles
            await self._detect_two_level_structure(root, agents_data)
        
        return agents_data
    
    async def _detect_three_level_structure(self, root: Path, agents_data: Dict[str, Any]):
        """Detects 3-level structure: Coordinators > Squads > Agents"""
        
        for level_2_dir in root.iterdir():
            if not level_2_dir.is_dir():
                continue
            
            dir_name = level_2_dir.name.lower()
            
            # Level 2: Detect if it's coordinator or squad
            if any(coord in dir_name for coord in self.detection_patterns["structure_keywords"]["coordinators"]):
                # It's a coordinator
                coordinator = await self._create_detected_agent(level_2_dir, "coordinator")
                agents_data["coordinators"].append(coordinator)
                agents_data["total_count"] += 1
                
            elif level_2_dir.is_dir() and any(level_2_dir.iterdir()):
                # It's a squad with agents
                squad = await self._create_detected_squad(level_2_dir)
                agents_data["squads"].append(squad)
                agents_data["total_count"] += len(squad.agents)
    
    async def _detect_two_level_structure(self, root: Path, agents_data: Dict[str, Any]):
        """Detects 2-level structure: Coordinator > Agents"""
        
        for subdir in root.iterdir():
            if not subdir.is_dir():
                continue
            
            dir_name = subdir.name.lower()
            
            # Detect role based on name and content
            if any(coord in dir_name for coord in self.detection_patterns["structure_keywords"]["coordinators"]):
                coordinator = await self._create_detected_agent(subdir, "coordinator")
                agents_data["coordinators"].append(coordinator)
            else:
                agent = await self._create_detected_agent(subdir, "agent")
                agents_data["individual_agents"].append(agent)
            
            agents_data["total_count"] += 1
    
    async def _create_detected_agent(self, agent_path: Path, role: str) -> DetectedAgent:
        """Creates a detected agent from a directory"""
        
        # Search for configuration files
        config_file = None
        launch_script = None
        
        try:
            for file in agent_path.iterdir():
                if file.is_file():
                    filename = file.name.lower()
                    
                    if any(config in filename for config in self.detection_patterns["config_files"]):
                        config_file = file
                    
                    if any(script in filename for script in self.detection_patterns["launch_scripts"]):
                        launch_script = file
        except:
            pass
        
        # Extract capabilities from name
        capabilities = self._extract_capabilities_from_name(agent_path.name)
        
        # Create metadata
        metadata = {
            "has_config": config_file is not None,
            "has_launcher": launch_script is not None,
            "directory_size": self._get_directory_size(agent_path)
        }
        
        return DetectedAgent(
            id=self._generate_agent_id(agent_path),
            name=agent_path.name,
            path=agent_path,
            role=role,
            capabilities=capabilities,
            config_file=config_file,
            launch_script=launch_script,
            metadata=metadata
        )
    
    async def _create_detected_squad(self, squad_path: Path) -> DetectedSquad:
        """Creates a detected squad from a directory"""
        
        agents = []
        leader_agents = []
        
        # Detect agents within the squad
        try:
            for subdir in squad_path.iterdir():
                if subdir.is_dir():
                    # Determine if it's leader or regular agent
                    dir_name = subdir.name.lower()
                    if any(leader in dir_name for leader in self.detection_patterns["structure_keywords"]["leaders"]):
                        agent = await self._create_detected_agent(subdir, "leader")
                        leader_agents.append(agent.id)
                    else:
                        agent = await self._create_detected_agent(subdir, "agent")
                    
                    agents.append(agent)
        except:
            pass
        
        # Search for squad configuration
        config_file = None
        try:
            for file in squad_path.iterdir():
                if file.is_file() and any(config in file.name.lower() for config in self.detection_patterns["config_files"]):
                    config_file = file
                    break
        except:
            pass
        
        # Extract purpose from name
        purpose = self._extract_purpose_from_name(squad_path.name)
        
        return DetectedSquad(
            id=self._generate_squad_id(squad_path),
            name=squad_path.name,
            path=squad_path,
            purpose=purpose,
            agents=agents,
            leader_agents=leader_agents,
            config_file=config_file,
            metadata={"agent_count": len(agents), "has_leaders": len(leader_agents) > 0}
        )
    
    async def _detect_configurations(self, root: Path) -> Dict[str, Any]:
        """Detects configuration files and technologies used"""
        
        config_data = {
            "config_files": [],
            "technologies": [],
            "languages": [],
            "frameworks": []
        }
        
        # Recursively scan configuration files
        for file_path in root.rglob("*"):
            if file_path.is_file():
                filename = file_path.name.lower()
                file_ext = file_path.suffix.lower()
                
                # Detect configuration files
                if any(config in filename for config in self.detection_patterns["config_files"]):
                    config_data["config_files"].append(file_path)
                
                # Detect technologies
                for tech, indicators in self.file_indicators.items():
                    if any(indicator in filename or indicator == file_ext for indicator in indicators):
                        if tech not in config_data["technologies"]:
                            config_data["technologies"].append(tech)
                
                # Detect languages
                if file_ext in [".py"]:
                    if "python" not in config_data["languages"]:
                        config_data["languages"].append("python")
                elif file_ext in [".js", ".ts"]:
                    if "javascript" not in config_data["languages"]:
                        config_data["languages"].append("javascript")
                elif file_ext in [".go"]:
                    if "go" not in config_data["languages"]:
                        config_data["languages"].append("go")
        
        return config_data
    
    def _calculate_detection_confidence(self, initial_analysis: Dict[str, Any], 
                                      structure_data: Dict[str, Any], 
                                      agents_data: Dict[str, Any]) -> float:
        """Calculates detection confidence (0.0-1.0)"""
        
        confidence_factors = []
        
        # Factor 1: Agent indicators found
        if agents_data["total_count"] > 0:
            confidence_factors.append(0.3)
        
        # Factor 2: Clear organizational structure
        if structure_data["organization_pattern"] != "flat":
            confidence_factors.append(0.2)
        
        # Factor 3: Configuration files present
        if initial_analysis["total_files"] > 0:
            confidence_factors.append(0.2)
        
        # Factor 4: Consistent nomenclature
        if len(initial_analysis["potential_agents"]) >= 2:
            confidence_factors.append(0.2)
        
        # Factor 5: Squads detected
        if len(agents_data["squads"]) > 0:
            confidence_factors.append(0.1)
        
        return sum(confidence_factors)
    
    def _extract_capabilities_from_name(self, name: str) -> List[str]:
        """Extracts capabilities from agent name"""
        
        capabilities = []
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        
        # Common capability patterns
        capability_patterns = {
            "web": ["web", "frontend", "backend", "fullstack"],
            "ai": ["ai", "ml", "intelligence", "smart"],
            "data": ["data", "analytics", "metrics"],
            "design": ["design", "ui", "ux", "graphic"],
            "security": ["security", "auth", "secure"],
            "testing": ["test", "qa", "quality"],
            "devops": ["devops", "deploy", "infrastructure"],
            "management": ["manager", "coordinator", "orchestrator"]
        }
        
        for capability, patterns in capability_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                capabilities.append(capability)
        
        return capabilities
    
    def _extract_purpose_from_name(self, name: str) -> str:
        """Extracts purpose from squad name"""
        
        name_clean = name.replace("_", " ").replace("-", " ").title()
        
        # Purpose patterns
        if "development" in name.lower() or "dev" in name.lower():
            return f"Software development - {name_clean}"
        elif "design" in name.lower():
            return f"Design and UX - {name_clean}"
        elif "data" in name.lower():
            return f"Data analysis - {name_clean}"
        elif "security" in name.lower():
            return f"Security and compliance - {name_clean}"
        elif "marketing" in name.lower():
            return f"Marketing and growth - {name_clean}"
        else:
            return f"Specialized team - {name_clean}"
    
    def _generate_project_id(self, root: Path) -> str:
        """Generates unique ID for the project"""
        return f"project_{root.name.lower().replace(' ', '_').replace('-', '_')}"
    
    def _generate_agent_id(self, agent_path: Path) -> str:
        """Generates unique ID for the agent"""
        return agent_path.name.lower().replace(" ", "_").replace("-", "_")
    
    def _generate_squad_id(self, squad_path: Path) -> str:
        """Generates unique ID for the squad"""
        return squad_path.name.lower().replace(" ", "_").replace("-", "_")
    
    def _get_directory_size(self, directory: Path) -> int:
        """Gets approximate directory size"""
        try:
            return len(list(directory.iterdir()))
        except:
            return 0

# Utility function for testing
async def test_detection(project_path: str):
    """Tests detection on a project"""
    
    detector = UniversalProjectDetector()
    detected = await detector.detect_project_structure(project_path)
    
    print(f"\nDETECTED PROJECT:")
    print(f"   Name: {detected.name}")
    print(f"   Type: {detected.project_type.value}")
    print(f"   Depth: {detected.structure_depth.value}")
    print(f"   Confidence: {detected.detection_confidence*100:.1f}%")
    
    print(f"\nAGENTS ({detected.total_agents}):")
    for coord in detected.coordinators:
        print(f"   Coordinator: {coord.name} ({', '.join(coord.capabilities)})")
    
    for squad in detected.squads:
        print(f"   Squad: {squad.name} ({len(squad.agents)} agents)")
        for agent in squad.agents[:3]:  # Show first 3
            print(f"      - {agent.name}")
        if len(squad.agents) > 3:
            print(f"      ... and {len(squad.agents)-3} more")
    
    for agent in detected.individual_agents[:5]:  # Show first 5
        print(f"   Agent: {agent.name}")
    
    if len(detected.individual_agents) > 5:
        print(f"   ... and {len(detected.individual_agents)-5} more agents")
    
    return detected

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
        asyncio.run(test_detection(project_path))
    else:
        print("Usage: python universal_project_detector.py /path/to/project")
        print("Example: python universal_project_detector.py /mnt/c/Users/pedro/Desktop/agente/Diamond-Team-Clean")