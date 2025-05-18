from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pathlib import Path
from personalized_narratives.tools.exa_search_tool import ExaSearchTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class PersonalizedNarratives:
    """PersonalizedNarratives crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    article_output_path: Path

    def __init__(self, article_output_path: Path = Path(".")):
        """Initialize with an article-specific output path."""
        self.article_output_path = article_output_path
        # Ensure the base for task outputs exists (e.g., output/news_N123/outputs/)
        # The tasks will write into this directory.
        (self.article_output_path / "outputs").mkdir(parents=True, exist_ok=True)

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def topic_extractor(self) -> Agent:
        return Agent(config=self.agents_config["topic_extractor"], verbose=True)

    @agent
    def news_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config["news_retriever"],
            tools=[ExaSearchTool()],
            verbose=True
        )

    @agent
    def relevance_filter(self) -> Agent:
        return Agent(config=self.agents_config["relevance_filter"], verbose=True)

    @agent
    def background_synthesizer(self) -> Agent:
        return Agent(config=self.agents_config["background_synthesizer"], verbose=True)

    @agent
    def storytelling_news_composer(self) -> Agent:
        return Agent(config=self.agents_config["storytelling_news_composer"], verbose=True)

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def extract_topics_entities(self) -> Task:
        return Task(
            config=self.tasks_config["extract_topics_entities"],  # type: ignore[index]
            output_file=str(self.article_output_path / "outputs" / "topics_entities.json"),
            inputs={"title": "{title}", "abstract": "{abstract}"},
        )

    @task
    def retrieve_candidate_articles(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_candidate_articles"],
            output_file=str(self.article_output_path / "outputs" / "exa_results.json"),
            inputs={"publication_date": "{publication_date}"},
        )

    @task
    def filter_and_rank_articles(self) -> Task:
        return Task(
            config=self.tasks_config["filter_and_rank_articles"],  # type: ignore[index]
            output_file=str(self.article_output_path / "outputs" / "ranked_articles.json"),
        )

    @task
    def synthesize_background(self) -> Task:
        return Task(
            config=self.tasks_config["synthesize_background"],  # type: ignore[index]
            output_file=str(self.article_output_path / "outputs" / "background_section.md"),
        )

    @task
    def compose_storytelling_news(self) -> Task:
        return Task(
            config=self.tasks_config["compose_storytelling_news"],  # type: ignore[index]
            output_file=str(self.article_output_path / "outputs" / "storytelling_news.md"),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PersonalizedNarratives crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=[
                self.topic_extractor(),
                self.news_retriever(),
                self.relevance_filter(),
                self.background_synthesizer(),
                self.storytelling_news_composer(),
            ],
            tasks=[
                self.extract_topics_entities(),
                self.retrieve_candidate_articles(),
                self.filter_and_rank_articles(),
                self.synthesize_background(),
                self.compose_storytelling_news(),
            ],
            process=Process.sequential,
            verbose=True,
        )
