import argparse

from datasets import Dataset

from prepare_canary import (
    aggregate_rows,
    arxiv_metadata,
    canonical_arxiv_id,
    canonical_url,
    parse_output_raw,
    stratified_limit,
)


def test_trace_parser_normalizes_arxiv_and_urls():
    assert canonical_arxiv_id("2501.01540v2") == "2501.01540"
    assert canonical_arxiv_id("1") is None
    assert canonical_url("http://www.github.com/Owner/Repo/tree/main") == "https://github.com/Owner/Repo"
    assert parse_output_raw(
        'reasoning\n```json\n{"github_url":"https://github.com/Owner/Repo","project_page_url":""}\n```'
    ) == ("https://github.com/Owner/Repo", None)


def test_arxiv_metadata_parses_atom(monkeypatch):
    atom = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <title>A  useful\n paper</title><summary>An abstract.</summary>
    </entry></feed>"""

    class Response:
        text = atom

        def raise_for_status(self):
            return None

    monkeypatch.setattr("prepare_canary.httpx.get", lambda *args, **kwargs: Response())
    assert arxiv_metadata("2501.00001") == ("A useful paper", "An abstract.")


def test_aggregate_excludes_seen_ids_and_requires_consensus():
    rows = Dataset.from_list(
        [
            {
                "arxiv_id": arxiv_id,
                "output_raw": f'```json\n{{"github_url":{github},"project_page_url":null}}\n```',
            }
            for arxiv_id, github in [
                ("2501.00001", '"https://github.com/example/seen"'),
                ("2501.00002", '"https://github.com/example/repo"'),
                ("2501.00002v2", '"https://github.com/example/repo"'),
                ("2501.00002", '"https://github.com/example/repo"'),
                ("1", '"https://github.com/example/bad"'),
            ]
        ]
    )
    args = argparse.Namespace(min_votes=3, min_agreement=0.6)
    aggregated = aggregate_rows(rows, {"2501.00001"}, args)
    assert [row["arxiv_id"] for row in aggregated] == ["2501.00002"]
    assert aggregated[0]["github"]["votes"] == 3
    assert aggregated[0]["project_page"]["value"] is None


def test_stratified_limit_covers_each_url_shape():
    rows = []
    for index, (github, project) in enumerate([("g", "p"), ("g", None), (None, "p"), (None, None), ("g2", "p2")]):
        rows.append(
            {
                "arxiv_id": f"2501.{index:05d}",
                "github": {"value": github},
                "project_page": {"value": project},
            }
        )
    assert [row["arxiv_id"] for row in stratified_limit(rows, 4)] == [
        "2501.00000",
        "2501.00001",
        "2501.00002",
        "2501.00003",
    ]
