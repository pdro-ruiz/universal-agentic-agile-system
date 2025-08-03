#!/usr/bin/env python3
"""
Deep Agile Methodology - Deep and Complete Agile Methodology
Real implementation of Scrum/Agile with all advanced components
"""

import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import uuid
from abc import ABC, abstractmethod

# ========== CORE AGILE ENTITIES ==========

class Priority(Enum):
    CRITICAL = 1
    HIGH = 2  
    MEDIUM = 3
    LOW = 4

class StoryPoints(Enum):
    XS = 1    # 1 story point
    S = 2     # 2 story points  
    M = 3     # 3 story points
    L = 5     # 5 story points
    XL = 8    # 8 story points
    XXL = 13  # 13 story points
    EPIC = 21 # 21+ story points

class UserStoryStatus(Enum):
    DRAFT = "draft"
    READY = "ready"              # Definition of Ready met
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    TESTING = "testing"
    DONE = "done"               # Definition of Done met
    ACCEPTED = "accepted"       # Product Owner accepted

class EpicStatus(Enum):
    CONCEPT = "concept"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

@dataclass
class AcceptanceCriteria:
    """Acceptance criteria for User Stories"""
    id: str
    description: str
    given: str      # Given - initial context
    when: str       # When - action that is executed  
    then: str       # Then - expected result
    priority: Priority = Priority.MEDIUM
    is_met: bool = False
    tested_by: Optional[str] = None
    test_evidence: Optional[str] = None

@dataclass
class DefinitionOfDone:
    """Definition of Done for the team"""
    id: str
    criteria: List[str]
    applies_to: List[str]  # ["user_story", "epic", "sprint"]
    mandatory: bool = True
    created_by: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    
    def is_satisfied(self, checklist: Dict[str, bool]) -> bool:
        """Checks if the Definition of Done is satisfied"""
        if not self.mandatory:
            return True
        
        return all(checklist.get(criterion, False) for criterion in self.criteria)

@dataclass
class UserStory:
    """Complete User Story with deep agile methodology"""
    id: str
    title: str
    description: str
    
    # Agile structure: As a [who], I want [what], so that [why]
    as_a: str           # Who - user type
    i_want: str         # What - desired functionality
    so_that: str        # Why - value/benefit
    
    # Story details
    priority: Priority = Priority.MEDIUM
    story_points: StoryPoints = StoryPoints.M
    status: UserStoryStatus = UserStoryStatus.DRAFT
    
    # Acceptance criteria & DoD
    acceptance_criteria: List[AcceptanceCriteria] = field(default_factory=list)
    definition_of_done: Optional[DefinitionOfDone] = None
    
    # Sprint & Epic association
    epic_id: Optional[str] = None
    sprint_id: Optional[str] = None
    
    # Assignment & tracking
    assigned_to: Optional[str] = None
    created_by: str = ""
    product_owner: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    business_value: int = 0  # 1-100
    technical_risk: int = 0  # 1-100
    dependencies: List[str] = field(default_factory=list)
    
    def is_ready_for_sprint(self) -> bool:
        """Checks Definition of Ready"""
        return (
            bool(self.as_a and self.i_want and self.so_that) and
            len(self.acceptance_criteria) > 0 and
            self.story_points != StoryPoints.EPIC and
            self.priority != Priority.CRITICAL or self.business_value > 0
        )
    
    def is_done(self) -> bool:
        """Checks if it is completely finished"""
        if not self.definition_of_done:
            return self.status == UserStoryStatus.DONE
        
        # Check acceptance criteria
        criteria_met = all(ac.is_met for ac in self.acceptance_criteria)
        
        # Check Definition of Done (simplified)
        dod_checklist = {
            "code_reviewed": True,
            "tests_pass": True,
            "documentation_updated": True,
            "deployed_to_staging": True
        }
        
        return criteria_met and self.definition_of_done.is_satisfied(dod_checklist)

@dataclass
class Epic:
    """Epic - collection of related User Stories"""
    id: str
    title: str
    description: str
    goal: str           # Business objective of the Epic
    
    status: EpicStatus = EpicStatus.CONCEPT
    priority: Priority = Priority.MEDIUM
    
    # Associated User Stories
    user_stories: List[str] = field(default_factory=list)  # IDs of user stories
    
    # Business context
    business_value: int = 0     # 1-100
    target_audience: str = ""
    success_metrics: List[str] = field(default_factory=list)
    
    # Planning
    estimated_story_points: int = 0
    target_completion: Optional[datetime] = None
    
    # Ownership
    product_owner: str = ""
    stakeholders: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def get_completion_percentage(self, stories: Dict[str, UserStory]) -> float:
        """Calculates Epic completion percentage"""
        if not self.user_stories:
            return 0.0
        
        completed_stories = sum(
            1 for story_id in self.user_stories 
            if story_id in stories and stories[story_id].status == UserStoryStatus.DONE
        )
        
        return (completed_stories / len(self.user_stories)) * 100

@dataclass
class ProductBacklogItem:
    """Product Backlog item with advanced prioritization"""
    id: str
    user_story_id: str
    priority_score: float = 0.0    # Automatically calculated
    
    # Prioritization factors (RICE framework)
    reach: int = 0          # Number of users impacted
    impact: int = 0         # Impact per user (1-3)
    confidence: int = 0     # Confidence in estimates (1-100%)
    effort: int = 0         # Effort in story points
    
    # MoSCoW prioritization
    moscow: str = "Should"  # Must, Should, Could, Won't
    
    # Kano model classification
    kano_category: str = "Performance"  # Basic, Performance, Excitement
    
    def calculate_priority_score(self) -> float:
        """Calculates priority score using RICE"""
        if self.effort == 0:
            return 0.0
        
        rice_score = (self.reach * self.impact * (self.confidence / 100)) / self.effort
        
        # Adjust by MoSCoW
        moscow_multiplier = {
            "Must": 2.0,
            "Should": 1.5, 
            "Could": 1.0,
            "Won't": 0.1
        }
        
        self.priority_score = rice_score * moscow_multiplier.get(self.moscow, 1.0)
        return self.priority_score

# ========== ADVANCED AGILE PROCESSES ==========

class BacklogGroomingSession:
    """Product Backlog refinement session"""
    
    def __init__(self, product_owner: str, scrum_master: str, team: List[str]):
        self.id = str(uuid.uuid4())
        self.product_owner = product_owner
        self.scrum_master = scrum_master
        self.team = team
        self.datetime = datetime.now()
        self.duration_minutes = 0
        
        self.refined_stories: List[str] = []
        self.estimated_stories: List[str] = []
        self.split_stories: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.removed_stories: List[str] = []
        
    async def refine_user_story(self, story: UserStory) -> UserStory:
        """Refines a User Story during grooming"""
        
        # 1. Check story clarity
        if not (story.as_a and story.i_want and story.so_that):
            print(f"Warning: Story {story.id} needs format clarification")
        
        # 2. Add/refine acceptance criteria
        if len(story.acceptance_criteria) == 0:
            await self._add_default_acceptance_criteria(story)
        
        # 3. Evaluate if split is needed
        if story.story_points == StoryPoints.EPIC or story.story_points.value > 8:
            print(f"Note: Story {story.id} is too large, consider splitting")
        
        # 4. Check dependencies
        await self._analyze_dependencies(story)
        
        self.refined_stories.append(story.id)
        return story
    
    async def _add_default_acceptance_criteria(self, story: UserStory):
        """Adds default acceptance criteria"""
        
        default_ac = AcceptanceCriteria(
            id=str(uuid.uuid4()),
            description=f"Basic criteria for {story.title}",
            given="The user is in the system",
            when=f"Executes the action: {story.i_want}",
            then=f"Gets the expected result: {story.so_that}"
        )
        
        story.acceptance_criteria.append(default_ac)

    async def _analyze_dependencies(self, story: UserStory):
        """Analyzes User Story dependencies"""
        
        # Automatic dependency analysis based on tags/keywords
        potential_deps = []
        
        story_text = f"{story.title} {story.description} {story.i_want}".lower()
        
        # Common dependency patterns
        if "authentication" in story_text or "login" in story_text:
            potential_deps.append("auth_system")
        
        if "database" in story_text or "data" in story_text:
            potential_deps.append("database_setup")
        
        if "api" in story_text:
            potential_deps.append("api_infrastructure")
        
        story.dependencies.extend(potential_deps)

class PlanningPokerSession:
    """Planning Poker session for estimation"""
    
    def __init__(self, facilitator: str, team: List[str]):
        self.id = str(uuid.uuid4())
        self.facilitator = facilitator
        self.team = team
        self.datetime = datetime.now()
        
        self.story_estimates: Dict[str, Dict[str, int]] = {}  # story_id -> {member: estimate}
        self.consensus_reached: Dict[str, bool] = {}
        self.final_estimates: Dict[str, StoryPoints] = {}
        
    async def estimate_story(self, story: UserStory, team_estimates: Dict[str, int]) -> StoryPoints:
        """Estimates a User Story using Planning Poker"""
        
        self.story_estimates[story.id] = team_estimates.copy()
        
        # Analyze estimates
        estimates = list(team_estimates.values())
        min_estimate = min(estimates)
        max_estimate = max(estimates)
        avg_estimate = sum(estimates) / len(estimates)
        
        # Check consensus (difference < 50%)
        variance_threshold = max_estimate * 0.5
        has_consensus = (max_estimate - min_estimate) <= variance_threshold
        
        self.consensus_reached[story.id] = has_consensus
        
        if has_consensus:
            # Use average estimate
            final_estimate = self._map_to_story_points(round(avg_estimate))
        else:
            # Needs more discussion
            print(f"Story {story.id} needs more discussion")
            print(f"   Estimates: min={min_estimate}, max={max_estimate}")
            
            # Use conservative estimate (maximum)
            final_estimate = self._map_to_story_points(max_estimate)
        
        self.final_estimates[story.id] = final_estimate
        story.story_points = final_estimate
        
        return final_estimate
    
    def _map_to_story_points(self, estimate: int) -> StoryPoints:
        """Maps numeric estimate to StoryPoints enum"""
        if estimate <= 1:
            return StoryPoints.XS
        elif estimate <= 2:
            return StoryPoints.S
        elif estimate <= 3:
            return StoryPoints.M
        elif estimate <= 5:
            return StoryPoints.L
        elif estimate <= 8:
            return StoryPoints.XL
        elif estimate <= 13:
            return StoryPoints.XXL
        else:
            return StoryPoints.EPIC

class ProductBacklog:
    """Product Backlog with advanced management"""
    
    def __init__(self, product_owner: str):
        self.product_owner = product_owner
        self.items: Dict[str, ProductBacklogItem] = {}
        self.user_stories: Dict[str, UserStory] = {}
        self.epics: Dict[str, Epic] = {}
        
        self.last_groomed: Optional[datetime] = None
        self.auto_prioritization_enabled = True
        
    def add_user_story(self, user_story: UserStory) -> ProductBacklogItem:
        """Adds User Story to Product Backlog"""
        
        self.user_stories[user_story.id] = user_story
        
        backlog_item = ProductBacklogItem(
            id=str(uuid.uuid4()),
            user_story_id=user_story.id
        )
        
        # Auto-populate prioritization data from story
        backlog_item.effort = user_story.story_points.value
        
        # Heuristic for reach/impact based on business_value
        if user_story.business_value > 80:
            backlog_item.reach = 1000
            backlog_item.impact = 3
            backlog_item.moscow = "Must"
        elif user_story.business_value > 60:
            backlog_item.reach = 500
            backlog_item.impact = 2
            backlog_item.moscow = "Should"
        else:
            backlog_item.reach = 100
            backlog_item.impact = 1
            backlog_item.moscow = "Could"
        
        # Confidence based on story clarity
        if user_story.is_ready_for_sprint():
            backlog_item.confidence = 90
        else:
            backlog_item.confidence = 50
        
        backlog_item.calculate_priority_score()
        self.items[backlog_item.id] = backlog_item
        
        return backlog_item
    
    def get_prioritized_backlog(self) -> List[ProductBacklogItem]:
        """Gets backlog prioritized by score"""
        
        if self.auto_prioritization_enabled:
            self._auto_prioritize()
        
        return sorted(
            self.items.values(),
            key=lambda item: item.priority_score,
            reverse=True
        )
    
    def _auto_prioritize(self):
        """Auto-prioritization based on algorithms"""
        
        for item in self.items.values():
            # Recalculate scores
            item.calculate_priority_score()
            
            # Adjust by age (older items get boost)
            story = self.user_stories.get(item.user_story_id)
            if story:
                days_old = (datetime.now() - story.created_at).days
                age_boost = min(days_old * 0.1, 1.0)  # Max 1.0 boost
                item.priority_score += age_boost
    
    async def groom_backlog(self, scrum_master: str, team: List[str]) -> BacklogGroomingSession:
        """Performs backlog grooming session"""
        
        session = BacklogGroomingSession(self.product_owner, scrum_master, team)
        
        # Select top stories for grooming
        prioritized_items = self.get_prioritized_backlog()
        top_stories = prioritized_items[:10]  # Top 10
        
        for item in top_stories:
            story = self.user_stories.get(item.user_story_id)
            if story and story.status == UserStoryStatus.DRAFT:
                refined_story = await session.refine_user_story(story)
                self.user_stories[story.id] = refined_story
        
        self.last_groomed = datetime.now()
        session.duration_minutes = 60  # Typically 1 hour
        
        return session

class SprintPlanning:
    """Sprint Planning with complete methodology"""
    
    def __init__(self, scrum_master: str, product_owner: str, team: List[str]):
        self.scrum_master = scrum_master
        self.product_owner = product_owner
        self.team = team
        self.datetime = datetime.now()
        
        self.sprint_goal: str = ""
        self.selected_stories: List[str] = []
        self.team_capacity: Dict[str, int] = {}  # member -> story points capacity
        self.commitment_points: int = 0
        
    async def plan_sprint(self, product_backlog: ProductBacklog, 
                         sprint_duration_days: int = 14) -> Dict[str, Any]:
        """Performs complete Sprint Planning"""
        
        planning_result = {
            "sprint_goal": "",
            "selected_stories": [],
            "total_story_points": 0,
            "team_capacity_used": 0.0,
            "planning_confidence": 0.0
        }
        
        # 1. Calculate team capacity
        total_capacity = await self._calculate_team_capacity(sprint_duration_days)
        
        # 2. Select sprint goal
        self.sprint_goal = await self._define_sprint_goal(product_backlog)
        
        # 3. Select User Stories
        selected_stories = await self._select_stories_for_sprint(
            product_backlog, total_capacity
        )
        
        # 4. Calculate metrics
        total_points = sum(
            product_backlog.user_stories[story_id].story_points.value 
            for story_id in selected_stories
        )
        
        capacity_used = total_points / total_capacity if total_capacity > 0 else 0
        
        # 5. Evaluate plan confidence
        confidence = await self._evaluate_planning_confidence(
            selected_stories, product_backlog
        )
        
        planning_result.update({
            "sprint_goal": self.sprint_goal,
            "selected_stories": selected_stories,
            "total_story_points": total_points,
            "team_capacity_used": capacity_used,
            "planning_confidence": confidence
        })
        
        self.selected_stories = selected_stories
        self.commitment_points = total_points
        
        return planning_result
    
    async def _calculate_team_capacity(self, sprint_days: int) -> int:
        """Calculates total team capacity in story points"""
        
        # Heuristic: each member can do ~2-3 story points per day
        points_per_member_per_day = 2.5
        working_days = sprint_days * 0.7  # Assuming 70% productive days
        
        base_capacity = len(self.team) * points_per_member_per_day * working_days
        
        # Adjust for meetings overhead, interruptions, etc.
        efficiency_factor = 0.8  # 80% efficiency
        
        total_capacity = int(base_capacity * efficiency_factor)
        
        # Save capacity per member
        capacity_per_member = total_capacity // len(self.team)
        for member in self.team:
            self.team_capacity[member] = capacity_per_member
        
        return total_capacity
    
    async def _define_sprint_goal(self, product_backlog: ProductBacklog) -> str:
        """Defines sprint goal based on top stories"""
        
        top_items = product_backlog.get_prioritized_backlog()[:5]
        
        # Analyze common themes in top stories
        themes = []
        for item in top_items:
            story = product_backlog.user_stories.get(item.user_story_id)
            if story:
                # Extract themes from tags and title
                themes.extend(story.tags)
                
                # Detect themes in title
                title_lower = story.title.lower()
                if "auth" in title_lower or "login" in title_lower:
                    themes.append("authentication")
                if "ui" in title_lower or "interface" in title_lower:
                    themes.append("user_interface")
                if "api" in title_lower:
                    themes.append("backend_api")
        
        # Most frequent theme
        if themes:
            most_common_theme = max(set(themes), key=themes.count)
            return f"Implement {most_common_theme} functionality"
        else:
            return "Deliver incremental value to the product"
    
    async def _select_stories_for_sprint(self, product_backlog: ProductBacklog, 
                                       capacity: int) -> List[str]:
        """Selects User Stories for the sprint"""
        
        selected = []
        used_capacity = 0
        
        prioritized_items = product_backlog.get_prioritized_backlog()
        
        for item in prioritized_items:
            story = product_backlog.user_stories.get(item.user_story_id)
            
            if not story or not story.is_ready_for_sprint():
                continue
            
            story_points = story.story_points.value
            
            # Check if it fits in remaining capacity
            if used_capacity + story_points <= capacity:
                selected.append(story.id)
                used_capacity += story_points
            
            # Stop if we reach near the limit
            if used_capacity >= capacity * 0.9:  # 90% of capacity
                break
        
        return selected
    
    async def _evaluate_planning_confidence(self, selected_stories: List[str], 
                                          product_backlog: ProductBacklog) -> float:
        """Evaluates confidence in the sprint plan"""
        
        if not selected_stories:
            return 0.0
        
        confidence_factors = []
        
        for story_id in selected_stories:
            story = product_backlog.user_stories.get(story_id)
            if story:
                # Factor 1: Story is well-defined
                if story.is_ready_for_sprint():
                    confidence_factors.append(0.3)
                
                # Factor 2: Has acceptance criteria
                if len(story.acceptance_criteria) > 0:
                    confidence_factors.append(0.2)
                
                # Factor 3: Not too large
                if story.story_points.value <= 8:
                    confidence_factors.append(0.2)
                
                # Factor 4: Few dependencies
                if len(story.dependencies) <= 2:
                    confidence_factors.append(0.2)
                
                # Factor 5: Clear business value
                if story.business_value > 0:
                    confidence_factors.append(0.1)
        
        # Average confidence
        avg_confidence = sum(confidence_factors) / len(selected_stories) if selected_stories else 0
        
        return min(avg_confidence, 1.0)

# ========== DEEP AGILE ORCHESTRATOR ==========

class DeepAgileOrchestrator:
    """Deep agile methodology orchestrator"""
    
    def __init__(self, product_owner: str, scrum_master: str, team: List[str]):
        self.product_owner = product_owner
        self.scrum_master = scrum_master
        self.team = team
        
        # Core components
        self.product_backlog = ProductBacklog(product_owner)
        self.epics: Dict[str, Epic] = {}
        
        # Definitions of Done per type
        self.definitions_of_done: Dict[str, DefinitionOfDone] = {}
        self._setup_default_definitions_of_done()
        
        # Sessions & activities
        self.grooming_sessions: List[BacklogGroomingSession] = []
        self.planning_poker_sessions: List[PlanningPokerSession] = []
        self.sprint_plannings: List[SprintPlanning] = []
        
    def _setup_default_definitions_of_done(self):
        """Sets up default Definitions of Done"""
        
        # DoD para User Stories
        story_dod = DefinitionOfDone(
            id="story_dod",
            criteria=[
                "code_reviewed",
                "unit_tests_written",
                "unit_tests_passing",
                "integration_tests_passing",
                "documentation_updated",
                "acceptance_criteria_met",
                "deployed_to_staging",
                "product_owner_approved"
            ],
            applies_to=["user_story"],
            mandatory=True,
            created_by=self.scrum_master
        )
        
        # DoD para Epics
        epic_dod = DefinitionOfDone(
            id="epic_dod",
            criteria=[
                "all_user_stories_completed",
                "end_to_end_tests_passing",
                "performance_requirements_met",
                "security_review_completed",
                "documentation_complete",
                "stakeholder_demo_approved",
                "deployed_to_production"
            ],
            applies_to=["epic"],
            mandatory=True,
            created_by=self.product_owner
        )
        
        self.definitions_of_done["user_story"] = story_dod
        self.definitions_of_done["epic"] = epic_dod
    
    async def create_epic(self, title: str, description: str, goal: str, 
                         business_value: int = 50) -> Epic:
        """Creates a complete Epic"""
        
        epic = Epic(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            goal=goal,
            business_value=business_value,
            product_owner=self.product_owner
        )
        
        self.epics[epic.id] = epic
        
        print(f"Epic created: {title}")
        print(f"   Goal: {goal}")
        print(f"   Business value: {business_value}/100")
        
        return epic
    
    async def create_user_story(self, title: str, as_a: str, i_want: str, so_that: str,
                              epic_id: Optional[str] = None, 
                              business_value: int = 50) -> UserStory:
        """Creates complete User Story"""
        
        story = UserStory(
            id=str(uuid.uuid4()),
            title=title,
            description=f"As {as_a}, I want {i_want}, so that {so_that}",
            as_a=as_a,
            i_want=i_want,
            so_that=so_that,
            epic_id=epic_id,
            business_value=business_value,
            created_by=self.product_owner,
            product_owner=self.product_owner,
            definition_of_done=self.definitions_of_done.get("user_story")
        )
        
        # Add to Product Backlog
        backlog_item = self.product_backlog.add_user_story(story)
        
        # Associate with Epic if specified
        if epic_id and epic_id in self.epics:
            self.epics[epic_id].user_stories.append(story.id)
        
        print(f"User Story created: {title}")
        print(f"   As: {as_a}")
        print(f"   I want: {i_want}")
        print(f"   So that: {so_that}")
        print(f"   Priority Score: {backlog_item.priority_score:.2f}")
        
        return story
    
    async def add_acceptance_criteria(self, story_id: str, description: str,
                                    given: str, when: str, then: str) -> AcceptanceCriteria:
        """Adds acceptance criteria to User Story"""
        
        if story_id not in self.product_backlog.user_stories:
            raise ValueError(f"User Story {story_id} not found")
        
        criteria = AcceptanceCriteria(
            id=str(uuid.uuid4()),
            description=description,
            given=given,
            when=when,
            then=then
        )
        
        self.product_backlog.user_stories[story_id].acceptance_criteria.append(criteria)
        
        print(f"Acceptance criteria added to {story_id}")
        
        return criteria
    
    async def conduct_backlog_grooming(self) -> BacklogGroomingSession:
        """Conducts backlog grooming session"""
        
        session = await self.product_backlog.groom_backlog(self.scrum_master, self.team)
        self.grooming_sessions.append(session)
        
        print(f"Backlog Grooming completed:")
        print(f"   {len(session.refined_stories)} stories refined")
        print(f"   Duration: {session.duration_minutes} minutes")
        
        return session
    
    async def conduct_planning_poker(self, story_ids: List[str]) -> PlanningPokerSession:
        """Conducts Planning Poker session"""
        
        session = PlanningPokerSession(self.scrum_master, self.team)
        
        for story_id in story_ids:
            if story_id in self.product_backlog.user_stories:
                story = self.product_backlog.user_stories[story_id]
                
                # Simulate team estimates (in real implementation would come from input)
                team_estimates = {
                    member: self._simulate_estimate(story) 
                    for member in self.team
                }
                
                final_estimate = await session.estimate_story(story, team_estimates)
                
                print(f"{story.title}: {final_estimate.name} ({final_estimate.value} pts)")
        
        self.planning_poker_sessions.append(session)
        
        return session
    
    def _simulate_estimate(self, story: UserStory) -> int:
        """Simulates estimation from a team member"""
        import random
        
        # Estimate based on perceived complexity
        base_estimate = 3  # Default
        
        # Adjust by business value (more value = more complex)
        if story.business_value > 80:
            base_estimate += 2
        elif story.business_value > 60:
            base_estimate += 1
        
        # Adjust by number of acceptance criteria
        base_estimate += len(story.acceptance_criteria)
        
        # Add variability (-1 to +2)
        variation = random.randint(-1, 2)
        
        return max(1, base_estimate + variation)
    
    async def plan_sprint(self, sprint_duration_days: int = 14) -> Dict[str, Any]:
        """Performs complete Sprint Planning"""
        
        planning = SprintPlanning(self.scrum_master, self.product_owner, self.team)
        result = await planning.plan_sprint(self.product_backlog, sprint_duration_days)
        
        self.sprint_plannings.append(planning)
        
        print(f"SPRINT PLANNING COMPLETED:")
        print(f"   Goal: {result['sprint_goal']}")
        print(f"   Story Points: {result['total_story_points']}")
        print(f"   Capacity used: {result['team_capacity_used']*100:.1f}%")
        print(f"   Confidence: {result['planning_confidence']*100:.1f}%")
        print(f"   {len(result['selected_stories'])} stories selected")
        
        return result
    
    async def get_backlog_health_metrics(self) -> Dict[str, Any]:
        """Gets Product Backlog health metrics"""
        
        total_stories = len(self.product_backlog.user_stories)
        ready_stories = sum(
            1 for story in self.product_backlog.user_stories.values()
            if story.is_ready_for_sprint()
        )
        
        total_epics = len(self.epics)
        completed_epics = sum(
            1 for epic in self.epics.values()
            if epic.status == EpicStatus.COMPLETED
        )
        
        # Calculate velocity trend (simplified)
        avg_story_points = sum(
            story.story_points.value 
            for story in self.product_backlog.user_stories.values()
        ) / total_stories if total_stories > 0 else 0
        
        metrics = {
            "total_stories": total_stories,
            "ready_stories": ready_stories,
            "ready_percentage": (ready_stories / total_stories * 100) if total_stories > 0 else 0,
            "total_epics": total_epics,
            "completed_epics": completed_epics,
            "epic_completion_rate": (completed_epics / total_epics * 100) if total_epics > 0 else 0,
            "avg_story_points": avg_story_points,
            "grooming_sessions": len(self.grooming_sessions),
            "planning_poker_sessions": len(self.planning_poker_sessions),
            "sprint_plannings": len(self.sprint_plannings),
            "last_grooming": self.product_backlog.last_groomed
        }
        
        return metrics

# Deep agile system demo
async def demo_deep_agile():
    """Deep agile methodology demo"""
    
    print("DEEP AGILE METHODOLOGY - DEMO")
    print("="*50)
    
    # Create team
    orchestrator = DeepAgileOrchestrator(
        product_owner="Maria_Product_Owner",
        scrum_master="Carlos_Scrum_Master", 
        team=["Ana_Developer", "Luis_Developer", "Sofia_QA", "Pedro_DevOps"]
    )
    
    # 1. Create Epic
    epic = await orchestrator.create_epic(
        title="Authentication System",
        description="Implement complete authentication and authorization system",
        goal="Allow users to register and access securely",
        business_value=90
    )
    
    # 2. Create User Stories for the Epic
    story1 = await orchestrator.create_user_story(
        title="User Registration",
        as_a="new user",
        i_want="to register in the system",
        so_that="I can access the functionalities",
        epic_id=epic.id,
        business_value=85
    )
    
    story2 = await orchestrator.create_user_story(
        title="User Login",
        as_a="registered user",
        i_want="to log in with my credentials",
        so_that="I can access my account",
        epic_id=epic.id,
        business_value=90
    )
    
    story3 = await orchestrator.create_user_story(
        title="Password Recovery",
        as_a="user",
        i_want="to recover my forgotten password",
        so_that="I can access my account again",
        epic_id=epic.id,
        business_value=60
    )
    
    # 3. Add Acceptance Criteria
    await orchestrator.add_acceptance_criteria(
        story1.id,
        "Unique email validation",
        given="A user attempts to register",
        when="Enters an already existing email",
        then="The system shows duplicate email error"
    )
    
    await orchestrator.add_acceptance_criteria(
        story2.id,
        "Successful login",
        given="A registered user",
        when="Enters correct credentials",
        then="Is redirected to the main dashboard"
    )
    
    # 4. Backlog Grooming
    grooming = await orchestrator.conduct_backlog_grooming()
    
    # 5. Planning Poker
    story_ids = [story1.id, story2.id, story3.id]
    poker = await orchestrator.conduct_planning_poker(story_ids)
    
    # 6. Sprint Planning
    sprint_plan = await orchestrator.plan_sprint(14)
    
    # 7. Health metrics
    metrics = await orchestrator.get_backlog_health_metrics()
    
    print(f"\nBACKLOG HEALTH METRICS:")
    print(f"   Total stories: {metrics['total_stories']}")
    print(f"   Ready stories: {metrics['ready_stories']} ({metrics['ready_percentage']:.1f}%)")
    print(f"   Completed epics: {metrics['completed_epics']}/{metrics['total_epics']}")
    print(f"   Average story points: {metrics['avg_story_points']:.1f}")
    
    return orchestrator

if __name__ == "__main__":
    asyncio.run(demo_deep_agile())