#!/usr/bin/env python3
"""
Two-Level Graph Prototype
Domain nodes: Math, Physics, English
Each domain has an A→B graph for reasoning.
Outer graph routes queries to relevant domains.
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import json


@dataclass
class Node:
    """A single A→B node (concept or reasoning step)"""
    id: str
    label: str
    domain: str


@dataclass
class Edge:
    """Connection between two nodes (A→B relationship)"""
    source: str
    target: str
    label: str


class DomainGraph:
    """Inner A→B graph for a single domain"""

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    def add_node(self, node_id: str, label: str):
        """Add a concept node"""
        self.nodes[node_id] = Node(node_id, label, self.domain_name)

    def add_edge(self, source: str, target: str, label: str):
        """Add a reasoning edge (A→B relationship)"""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Source or target node not found")
        self.edges.append(Edge(source, target, label))

    def reason(self, start_node: str = None) -> str:
        """
        Traverse the domain's A→B graph and build a reasoning chain.
        If no start_node specified, start from first node and follow edges.
        Returns the complete reasoning chain as text.
        """
        if not self.nodes:
            return f"[{self.domain_name}] Empty domain"

        # Use specified start or first node
        current = start_node or list(self.nodes.keys())[0]

        if current not in self.nodes:
            return f"[{self.domain_name}] Start node '{current}' not found"

        # Build adjacency list
        graph = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            graph[edge.source].append((edge.target, edge.label))

        # Follow chain from start node
        chain = [self.nodes[current].label]
        visited = {current}

        while current in graph and graph[current]:
            next_node, edge_label = graph[current][0]  # Follow first edge
            if next_node in visited:
                break
            chain.append(edge_label)
            chain.append(self.nodes[next_node].label)
            visited.add(next_node)
            current = next_node

        reasoning = " → ".join(chain)
        return f"[{self.domain_name}] {reasoning}"


class OuterGraph:
    """Outer routing graph that connects domain nodes"""

    def __init__(self):
        self.domains: Dict[str, DomainGraph] = {}
        self.domain_keywords: Dict[str, Set[str]] = {}

    def add_domain(self, domain_name: str, keywords: Set[str]) -> DomainGraph:
        """Register a domain and its trigger keywords"""
        domain = DomainGraph(domain_name)
        self.domains[domain_name] = domain
        self.domain_keywords[domain_name] = keywords
        return domain

    def route(self, query: str) -> Set[str]:
        """Determine which domains are relevant for a query"""
        query_lower = query.lower()
        relevant_domains = set()

        for domain_name, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    relevant_domains.add(domain_name)
                    break

        return relevant_domains if relevant_domains else {"Math"}  # Default to Math

    def query(self, user_query: str, start_node: str = None) -> str:
        """
        Process a query:
        1. Route to relevant domains
        2. Each domain reasons through its A→B graph
        3. Aggregate results
        """
        # Route
        relevant = self.route(user_query)

        print(f"\n{'='*60}")
        print(f"Query: {user_query}")
        print(f"Routed to: {', '.join(relevant)}")
        print(f"{'='*60}")

        # Reason in each domain
        results = []
        for domain_name in relevant:
            domain = self.domains[domain_name]
            reasoning = domain.reason(start_node)
            results.append(reasoning)
            print(reasoning)

        # Aggregate
        print(f"\n{'─'*60}")
        print("Aggregated Result:")
        aggregated = "\n".join(results) if results else "No domains activated"
        print(aggregated)
        print(f"{'='*60}\n")

        return aggregated


def setup_math_domain() -> DomainGraph:
    """Build the Math domain with A→B graph"""
    math = DomainGraph("Math")

    # Nodes
    math.add_node("num", "Number")
    math.add_node("op", "Operation")
    math.add_node("result", "Result")
    math.add_node("check", "Verification")

    # Edges (reasoning chain)
    math.add_edge("num", "op", "apply")
    math.add_edge("op", "result", "compute")
    math.add_edge("result", "check", "validate")

    return math


def setup_physics_domain() -> DomainGraph:
    """Build the Physics domain with A→B graph"""
    physics = DomainGraph("Physics")

    # Nodes
    physics.add_node("force", "Force")
    physics.add_node("mass", "Mass")
    physics.add_node("accel", "Acceleration")
    physics.add_node("motion", "Motion")
    physics.add_node("energy", "Energy")

    # Edges (reasoning chain: F=ma, then kinetic energy)
    physics.add_edge("force", "accel", "F=ma")
    physics.add_edge("mass", "accel", "F=ma")
    physics.add_edge("accel", "motion", "integrate")
    physics.add_edge("motion", "energy", "kinetic energy")

    return physics


def setup_english_domain() -> DomainGraph:
    """Build the English domain with A→B graph"""
    english = DomainGraph("English")

    # Nodes
    english.add_node("word", "Word")
    english.add_node("grammar", "Grammar")
    english.add_node("meaning", "Meaning")
    english.add_node("context", "Context")
    english.add_node("interpret", "Interpretation")

    # Edges (reasoning chain: word → grammar → meaning → context → full interpretation)
    english.add_edge("word", "grammar", "parse")
    english.add_edge("grammar", "meaning", "define")
    english.add_edge("meaning", "context", "situate")
    english.add_edge("context", "interpret", "conclude")

    return english


def print_help():
    """Print available commands"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║              Two-Level Graph - Interactive CLI                ║
╚════════════════════════════════════════════════════════════════╝

Commands:
  query <text>              - Send a query to the graph
  add_node <domain> <id> <label>  - Add a node to a domain
  add_edge <domain> <source> <target> <label>  - Add an edge
  list_domains              - Show all domains
  list_nodes <domain>       - Show nodes in a domain
  list_edges <domain>       - Show edges in a domain
  demo                      - Run demo queries
  help                      - Show this help
  exit                      - Exit the program

Examples:
  query How do I calculate force?
  add_node math x "Variable X"
  add_edge math x y "equals"
  list_nodes math
    """)


def handle_command(cmd: str, graph: OuterGraph):
    """Parse and execute a user command"""
    parts = cmd.strip().split(maxsplit=1)
    if not parts:
        return

    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command == "query":
        graph.query(args)

    elif command == "add_node":
        tokens = args.split(maxsplit=2)
        if len(tokens) < 3:
            print("Usage: add_node <domain> <id> <label>")
            return
        domain_name, node_id, label = tokens[0], tokens[1], tokens[2]
        if domain_name not in graph.domains:
            print(f"Domain '{domain_name}' not found")
            return
        graph.domains[domain_name].add_node(node_id, label)
        print(f"✓ Added node '{node_id}' to {domain_name}")

    elif command == "add_edge":
        tokens = args.split(maxsplit=3)
        if len(tokens) < 4:
            print("Usage: add_edge <domain> <source> <target> <label>")
            return
        domain_name, source, target, label = tokens[0], tokens[1], tokens[2], tokens[3]
        if domain_name not in graph.domains:
            print(f"Domain '{domain_name}' not found")
            return
        try:
            graph.domains[domain_name].add_edge(source, target, label)
            print(f"✓ Added edge {source} → {target} in {domain_name}")
        except ValueError as e:
            print(f"✗ Error: {e}")

    elif command == "list_domains":
        print("\nAvailable domains:")
        for domain_name in graph.domains.keys():
            print(f"  - {domain_name}")
        print()

    elif command == "list_nodes":
        domain_name = args.strip()
        if domain_name not in graph.domains:
            print(f"Domain '{domain_name}' not found")
            return
        domain = graph.domains[domain_name]
        print(f"\nNodes in {domain_name}:")
        for node_id, node in domain.nodes.items():
            print(f"  {node_id:12} → {node.label}")
        print()

    elif command == "list_edges":
        domain_name = args.strip()
        if domain_name not in graph.domains:
            print(f"Domain '{domain_name}' not found")
            return
        domain = graph.domains[domain_name]
        print(f"\nEdges in {domain_name}:")
        for edge in domain.edges:
            print(f"  {edge.source} --[{edge.label}]--> {edge.target}")
        print()

    elif command == "demo":
        queries = [
            "How do I calculate the force?",
            "What does grammar mean?",
            "Explain motion in physics",
            "Define the word energy",
        ]
        for query in queries:
            graph.query(query)

    elif command == "help":
        print_help()

    elif command == "exit":
        print("Goodbye!")
        exit(0)

    else:
        print(f"Unknown command: {command}")
        print("Type 'help' for available commands")


# ============================================================================
# RECONSTRUCTED, NOT RECOVERED, from here down. The packed source this file
# was unpacked from is corrupted/truncated starting mid-way through main()
# (right after this comment's original counterpart, "# Register domains with
# keywords") -- everything above this marker is the real, original recovered
# source; everything below is a good-faith completion based on what the rest
# of the file clearly implies (setup_math_domain()/setup_physics_domain()/
# setup_english_domain() are fully defined and unused otherwise; print_help()
# documents this exact CLI shape; the "demo" queries above imply a keyword
# set for routing). If you still have the original file, prefer it over this.
# ============================================================================

def main():
    """Interactive CLI mode"""

    # Build outer graph
    graph = OuterGraph()

    # Register domains with keywords, then attach each domain's prebuilt
    # A->B graph (the setup_*_domain() functions above were otherwise
    # never called from anywhere).
    graph.add_domain("Math", {"calculate", "number", "equation", "math", "compute", "solve"})
    graph.add_domain("Physics", {"force", "motion", "energy", "physics", "mass", "acceleration"})
    graph.add_domain("English", {"grammar", "word", "meaning", "context", "english", "language", "define"})

    graph.domains["Math"] = setup_math_domain()
    graph.domains["Physics"] = setup_physics_domain()
    graph.domains["English"] = setup_english_domain()

    print_help()

    while True:
        try:
            cmd = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        handle_command(cmd, graph)


if __name__ == "__main__":
    main()
