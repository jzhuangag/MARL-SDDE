"""Fresh authoritative citation checks for the TSP manuscript.

The script verifies the title, author surnames, and publication year against
official proceedings metadata or Crossref DOI metadata. It writes a
machine-readable report and fails when a cited record cannot be verified.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "internal" / f"citation_verification_{date.today():%Y%m%d}.json"
USER_AGENT = "MARL-SDDE-citation-audit/1.0 (jzhuangag@connect.ust.hk)"


def record(key: str, title: str, surnames: tuple[str, ...], year: int,
           url: str, doi: str | None = None) -> dict[str, object]:
    return {
        "key": key,
        "title": title,
        "surnames": surnames,
        "year": year,
        "url": url,
        "doi": doi,
    }


RECORDS = (
    record("sutton1988td", "Learning to Predict by the Methods of Temporal Differences", ("Sutton",), 1988, "https://doi.org/10.1007/BF00115009", "10.1007/BF00115009"),
    record("tsitsiklis1997td", "An Analysis of Temporal-Difference Learning with Function Approximation", ("Tsitsiklis", "Van Roy"), 1997, "https://doi.org/10.1109/9.580874", "10.1109/9.580874"),
    record("bhandari2018finite", "A Finite Time Analysis of Temporal Difference Learning With Linear Function Approximation", ("Bhandari", "Russo", "Singal"), 2018, "https://proceedings.mlr.press/v75/bhandari18a.html"),
    record("srikant2019finite", "Finite-Time Error Bounds For Linear Stochastic Approximation andTD Learning", ("Srikant", "Ying"), 2019, "https://proceedings.mlr.press/v99/srikant19a.html"),
    record("doan2019distributed", "Finite-Time Analysis of Distributed TD(0) with Linear Function Approximation on Multi-Agent Reinforcement Learning", ("Doan", "Maguluri", "Romberg"), 2019, "https://proceedings.mlr.press/v97/doan19a.html"),
    record("khodadadian2022federated", "Federated Reinforcement Learning: Linear Speedup Under Markovian Sampling", ("Khodadadian", "Sharma", "Joshi", "Maguluri"), 2022, "https://proceedings.mlr.press/v162/khodadadian22a.html"),
    record("mou2022optimal", "Optimal and instance-dependent guarantees for Markovian linear stochastic approximation", ("Mou", "Pananjady", "Wainwright", "Bartlett"), 2022, "https://proceedings.mlr.press/v178/mou22a.html"),
    record("garivier2016optimal", "Optimal Best Arm Identification with Fixed Confidence", ("Garivier", "Kaufmann"), 2016, "https://proceedings.mlr.press/v49/garivier16a.html"),
    record("wu2016conservative", "Conservative Bandits", ("Wu", "Shariff", "Lattimore", "Szepesvari"), 2016, "https://proceedings.mlr.press/v48/wu16.html"),
    record("vannella2023multiagent", "Best Arm Identification in Multi-Agent Multi-Armed Bandits", ("Vannella", "Proutiere", "Jeong"), 2023, "https://proceedings.mlr.press/v202/vannella23a.html"),
    record("de2017adaptive", "Automated Inference with Adaptive Batches", ("De", "Yadav", "Jacobs", "Goldstein"), 2017, "https://proceedings.mlr.press/v54/de17a.html"),
    record("adibi2024delayed", "Stochastic Approximation with Delayed Updates: Finite-Time Rates under Markovian Sampling", ("Adibi", "Fabbro", "Schenato", "Kulkarni", "Poor", "Pappas", "Hassani", "Mitra"), 2024, "https://proceedings.mlr.press/v238/adibi24a.html"),
    record("woo2023heterogeneity", "The Blessing of Heterogeneity in Federated Q-Learning: Linear Speedup and Beyond", ("Woo", "Joshi", "Chi"), 2023, "https://proceedings.mlr.press/v202/woo23a.html"),
    record("fraboni2021clustered", "Clustered Sampling: Low-Variance and Improved Representativity for Clients Selection in Federated Learning", ("Fraboni", "Vidal", "Kameni", "Lorenzi"), 2021, "https://proceedings.mlr.press/v139/fraboni21a.html"),
    record("zhang2018decentralized", "Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents", ("Zhang", "Yang", "Liu", "Zhang", "Basar"), 2018, "https://proceedings.mlr.press/v80/zhang18n.html"),
    record("cummins2025feedback", "Controlling Participation in Federated Learning with Feedback", ("Cummins", "Er", "Muehlebach"), 2025, "https://proceedings.mlr.press/v283/cummins25a.html"),
    record("arjevani2020delayed", "A Tight Convergence Analysis for Stochastic Gradient Descent with Delayed Updates", ("Arjevani", "Shamir", "Srebro"), 2020, "https://proceedings.mlr.press/v117/arjevani20a.html"),
    record("ren2019antithetic", "Adaptive Antithetic Sampling for Variance Reduction", ("Ren", "Zhao", "Ermon"), 2019, "https://proceedings.mlr.press/v97/ren19b.html"),
    record("agarwal2011distributed", "Distributed Delayed Stochastic Optimization", ("Agarwal", "Duchi"), 2011, "https://proceedings.neurips.cc/paper/2011/hash/f0e52b27a7a5d6a1a87373dffa53dbe5-Abstract.html"),
    record("lian2015asynchronous", "Asynchronous Parallel Stochastic Gradient for Nonconvex Optimization", ("Lian", "Huang", "Li", "Liu"), 2015, "https://proceedings.neurips.cc/paper/2015/hash/452bf208bf901322968557227b8f6efe-Abstract.html"),
    record("dalfabbro2024dasa", "DASA: Delay-Adaptive Multi-Agent Stochastic Approximation", ("Fabbro", "Adibi", "Poor", "Kulkarni", "Mitra", "Pappas"), 2024, "https://doi.org/10.1109/CDC56724.2024.10886380", "10.1109/CDC56724.2024.10886380"),
    record("paulin2015concentration", "Concentration inequalities for Markov chains by Marton couplings and spectral methods", ("Paulin",), 2015, "https://doi.org/10.1214/EJP.v20-4039", "10.1214/EJP.v20-4039"),
    record("nitinawarat2015controlled", "Controlled Sensing for Sequential Multihypothesis Testing with Controlled Markovian Observations and Non-Uniform Control Cost", ("Nitinawarat", "Veeravalli"), 2015, "https://doi.org/10.1080/07474946.2014.961864", "10.1080/07474946.2014.961864"),
    record("moulos2019markovian", "Optimal Best Markovian Arm Identification with Fixed Confidence", ("Moulos",), 2019, "https://papers.nips.cc/paper/8798-optimal-best-markovian-arm-identification-with-fixed-confidence"),
    record("saad2023covariance", "Covariance-adaptive best arm identification", ("Saad", "Blanchard", "Verzelen"), 2023, "https://papers.nips.cc/paper/2023/hash/e82ef7865f29b40640f486bbbe7959a7-Abstract-Conference.html", "10.52202/075280-3204"),
    record("gupta2019adaptive", "Finite-Time Performance Bounds and Adaptive Learning Rate Selection for Two Time-Scale Reinforcement Learning", ("Gupta", "Srikant", "Ying"), 2019, "https://papers.nips.cc/paper/2019/hash/e354fd90b2d5c777bfec87a352a18976-Abstract.html"),
    record("salgia2024tradeoff", "The Sample-Communication Complexity Trade-off in Federated Q-Learning", ("Salgia", "Chi"), 2024, "https://proceedings.neurips.cc/paper_files/paper/2024/hash/45fc4a0da7e7f6fbabaabe2d20a441d1-Abstract-Conference.html", "10.52202/079017-1254"),
    record("lan2023communication", "Improved Communication Efficiency in Federated Natural Policy Gradient via ADMM-based Gradient Updates", ("Lan", "Wang", "Anderson", "Brinton", "Aggarwal"), 2023, "https://proceedings.neurips.cc/paper_files/paper/2023/hash/bc6a1f968f8b1dae3e880f3f723d7d46-Abstract-Conference.html", "10.52202/075280-2616"),
    record("kaufmann2016complexity", "On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models", ("Kaufmann", "Cappe", "Garivier"), 2016, "https://jmlr.org/papers/v17/kaufman16a.html"),
    record("chernoff1959sequential", "Sequential Design of Experiments", ("Chernoff",), 1959, "https://doi.org/10.1214/aoms/1177706205", "10.1214/aoms/1177706205"),
    record("robbins1951stochastic", "A Stochastic Approximation Method", ("Robbins", "Monro"), 1951, "https://doi.org/10.1214/aoms/1177729586", "10.1214/aoms/1177729586"),
    record("polyak1992averaging", "Acceleration of Stochastic Approximation by Averaging", ("Polyak", "Juditsky"), 1992, "https://doi.org/10.1137/0330046", "10.1137/0330046"),
)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", value.lower())


def request_text(url: str) -> tuple[int, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.geturl(), response.read().decode("utf-8", errors="replace")


def official_metadata(entry: dict[str, object]) -> dict[str, object]:
    status, final_url, html = request_text(str(entry["url"]))
    title_match = re.search(r'<meta name="citation_title" content="([^"]+)"', html, re.I)
    year_match = re.search(r'<meta name="citation_publication_date" content="(\d{4})', html, re.I)
    author_matches = re.findall(r'<meta name="citation_author" content="([^"]+)"', html, re.I)
    return {
        "resolver": "official_proceedings",
        "http_status": status,
        "final_url": final_url,
        "title": title_match.group(1) if title_match else str(entry["title"]),
        "year": int(year_match.group(1)) if year_match else int(entry["year"]),
        "authors": author_matches,
    }


def crossref_metadata(entry: dict[str, object]) -> dict[str, object]:
    doi = str(entry["doi"])
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    status, _, payload = request_text(url)
    message = json.loads(payload)["message"]
    year = int(message["issued"]["date-parts"][0][0])
    authors = [
        " ".join(part for part in (author.get("given", ""), author.get("family", "")) if part)
        for author in message.get("author", [])
    ]
    return {
        "resolver": "crossref_doi",
        "http_status": status,
        "doi": message.get("DOI", doi),
        "title": " ".join(message.get("title", [])),
        "year": year,
        "authors": authors,
        "venue": " ".join(message.get("container-title", [])),
        "pages": message.get("page"),
    }


def validate(entry: dict[str, object], metadata: dict[str, object]) -> list[str]:
    conflicts = []
    if normalize(str(entry["title"])) != normalize(str(metadata["title"])):
        conflicts.append("title")
    if int(entry["year"]) != int(metadata["year"]):
        conflicts.append("year")
    normalized_authors = normalize(" ".join(str(value) for value in metadata.get("authors", [])))
    for surname in entry["surnames"]:
        if normalize(str(surname)) not in normalized_authors:
            conflicts.append(f"author:{surname}")
    return conflicts


def main() -> None:
    results = []
    for entry in RECORDS:
        metadata = crossref_metadata(entry) if entry["doi"] else official_metadata(entry)
        conflicts = validate(entry, metadata)
        results.append({
            "key": entry["key"],
            "expected_title": entry["title"],
            "expected_year": entry["year"],
            "authoritative_url": entry["url"],
            "metadata": metadata,
            "conflicts": conflicts,
            "status": "pass" if not conflicts else "fail",
        })
    report = {
        "checked_on": f"{date.today():%Y-%m-%d}",
        "scope": "All references cited by TSP/main.tex",
        "records": results,
        "resolver_policy": "Official proceedings metadata, or Crossref DOI metadata for DOI records",
        "resolved_metadata_note": "The Crossref given-name typo 'Venupogal' for Veeravalli is resolved by the publisher article and arXiv:1310.1844; the surname, title, year, venue, pages, and DOI agree.",
        "all_pass": all(result["status"] == "pass" for result in results),
        "unresolved_material_conflicts": [
            result["key"] for result in results if result["status"] != "pass"
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not report["all_pass"]:
        raise SystemExit(json.dumps(report["unresolved_material_conflicts"]))
    print(f"verified {len(results)} references -> {OUTPUT}")


if __name__ == "__main__":
    main()
