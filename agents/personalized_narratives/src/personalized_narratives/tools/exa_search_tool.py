# from crewai_tools import tool  # Removed, not available in your version
from exa_py import Exa
import os
import json
from datetime import datetime, timedelta
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class ExaSearchToolSchema(BaseModel):
    topics: list = Field(..., description="List of topics for the search")
    entities: list = Field(None, description="List of entities for the search")
    keywords: list = Field(None, description="List of keywords for the search")
    publication_date: str = Field(..., description="The end date (YYYY-MM-DD) for the search window")
    num_results: int = Field(15, description="Number of results to return")

class ExaSearchTool(BaseTool):
    name: str = "Exa search and get contents"
    description: str = (
        "Tool using Exa's Python SDK to run semantic search and return result highlights. "
        "topics, entities, keywords: lists of strings for query construction. "
        "publication_date: the date of the original article (YYYY-MM-DD)."
    )
    args_schema: Type[BaseModel] = ExaSearchToolSchema

    def _run(self, topics, entities=None, keywords=None, publication_date=None, num_results=15):
        exa_api_key = os.getenv("EXA_API_KEY")
        exa = Exa(exa_api_key)
        if entities is None:
            entities = topics
        if keywords is None:
            keywords = topics
        query = (
            f"Topics: [{', '.join(topics)}]\n"
            f"Entities: [{', '.join(entities)}]\n"
            f"Keywords: [{', '.join(keywords)}]"
        )
        if publication_date:
            end_dt = datetime.strptime(publication_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=30)
            end_date = end_dt.strftime("%Y-%m-%dT23:59:59.999Z")
            start_date = start_dt.strftime("%Y-%m-%dT00:00:00.000Z")
        else:
            end_date = None
            start_date = None
        response = exa.search_and_contents(
            query,
            type="auto",
            num_results=num_results,
            text={"max_characters": 500},
            start_published_date=start_date,
            end_published_date=end_date,
        )
        results = []
        for eachResult in response.results:
            result = {
                "title": eachResult.title,
                "url": eachResult.url,
                "published_date": getattr(eachResult, "published_date", None),
                "score": getattr(eachResult, "score", None),
                "source": getattr(eachResult, "source", None),
                "text": getattr(eachResult, "text", None),
            }
            results.append(result)
        return json.dumps(results, ensure_ascii=False, indent=2)
