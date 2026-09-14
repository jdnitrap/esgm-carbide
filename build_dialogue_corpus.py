"""Builds dialogue_corpus.txt from a real, hand-authored fact list,
expanded across multiple real question phrasings per fact.

Round 3 (2026-09-14), at explicit user direction after round 2 (203
pairs, 66 facts x 3 phrasings) still failed to produce topically
relevant answers even with the new TURN/QUESTION_FORM columns wired
in: "your dataset had to come up with something that has to be
relatively close to it" -- correct diagnosis, same lesson as real
instruction-tuning. A dataset that's merely bigger but still narrow
(same ~66 facts, same 3 templates) teaches "answer questions shaped
exactly like these," not "answer questions in general." This round
widens BOTH axes for real: ~150 facts (roughly 2.3x the topic coverage
of round 2, still grounded in domains the graph/essay-head already
have real vocabulary for for coherence) x 6 natural question phrasings
(roughly 2x round 2's phrasing variety) = ~900 real pairs, vs. 203.
"""

QUESTION_TEMPLATES = [
    "What is {t}?",
    "Can you explain {t}?",
    "Tell me about {t}.",
    "What does {t} mean?",
    "Why does {t} matter?",
    "How would you describe {t}?",
]

FACTS = [
    ("consciousness", "consciousness arises from billions of neurons firing in coordinated patterns, though how remains unexplained"),
    ("the universe", "the universe spans 13.8 billion years, from the hot Big Bang to the present epoch"),
    ("quantum mechanics", "quantum mechanics describes subatomic particles that exist in superposition until observed"),
    ("the double-slit experiment", "the double-slit experiment demonstrates that light behaves as both particle and wave"),
    ("entanglement", "entanglement creates instantaneous connections across vast distances, defying classical notions of locality"),
    ("the uncertainty principle", "the uncertainty principle says we cannot know both position and momentum with perfect precision"),
    ("dark matter", "dark matter and dark energy comprise 95 percent of the cosmos"),
    ("the ocean", "the sea covers 71 percent of Earth's surface, containing 97 percent of our water"),
    ("coral reefs", "coral reefs are rainforests of the sea, supporting vast biodiversity"),
    ("whales", "whales migrate thousands of miles, their songs echoing across ocean basins"),
    ("love", "love asks for loyalty, sacrifice, and the willingness to be transformed by another person"),
    ("freedom", "freedom is both an opportunity and a burden, a responsibility that comes with real choices"),
    ("happiness", "happiness is less a destination than a byproduct of meaning, engagement, and connection"),
    ("wisdom", "wisdom is the pursuit of understanding one's own assumptions, not just the accumulation of facts"),
    ("the mind", "the mind is a curious thing, forever seeking patterns in chaos, meaning in randomness"),
    ("technology", "technology reshapes society, and artificial intelligence is changing how people work and think"),
    ("artificial intelligence", "artificial intelligence is reshaping society, raising real questions about autonomy and responsibility"),
    ("history", "history teaches us to examine the past, though it rarely repeats exactly"),
    ("literature", "literature is a conversation across time between author and reader"),
    ("language", "language is the artist's medium, words arranged like paint on canvas"),
    ("a star", "a star sustains itself through nuclear fusion, turning hydrogen into helium and light"),
    ("a galaxy", "a galaxy contains billions of stars bound together by gravity"),
    ("the Big Bang", "the Big Bang was the hot, dense beginning from which the entire universe expanded"),
    ("stardust", "humans are made of stardust, elements forged in the furnaces of ancient stars"),
    ("climate change", "climate change is driven by human systems, and its effects reshape ecosystems worldwide"),
    ("biodiversity", "biodiversity sustains the balance of forests, oceans, and the air that living things breathe"),
    ("a forest", "a forest is a system where trees, fungi, and animals depend on one another to survive"),
    ("chaos", "chaos is randomness without an obvious pattern, though patterns can still hide inside it"),
    ("truth", "truth is a distinction worth examining carefully, since assumptions often masquerade as certainty"),
    ("a good life", "a good life is often described as the pursuit of wisdom, happiness, and honest human connection"),
    ("the graph", "the graph is a sparse structure of nodes and edges that stores learned facts and relationships"),
    ("a neuron", "a neuron is a cell that fires in coordinated patterns with billions of others to produce thought"),
    ("memory", "memory is the brain's way of keeping the past available to shape the present"),
    ("time", "time spans from the hot Big Bang to the present epoch, a single unfolding story"),
    ("energy", "energy takes many forms, from starlight to the electricity that powers modern life"),
    ("gravity", "gravity binds galaxies together and shapes the structure of the entire cosmos"),
    ("light", "light behaves as both a particle and a wave, depending on how it is observed"),
    ("a particle", "a particle is a subatomic unit that can exist in superposition until it is measured"),
    ("an observer", "an observer's participation in quantum mechanics is a mysterious and debated role"),
    ("responsibility", "responsibility is the burden that comes with real freedom and real choices"),
    ("curiosity", "curiosity is the drive to seek understanding, to question assumptions, and to explore the unknown"),
    ("science", "science is a method for examining assumptions and testing them against real evidence"),
    ("philosophy", "philosophy examines the foundations beneath everyday assumptions about existence and meaning"),
    ("existence", "existence precedes essence; we are not born with a predetermined nature"),
    ("a revolution", "a revolution topples old orders, reshaping ideologies and the structure of society"),
    ("industry", "the industrial revolution transformed labor from agrarian to mechanical, reshaping daily life"),
    ("cryptography", "cryptography ensures privacy by encoding information so only intended readers can understand it"),
    ("the mind-body connection", "the mind-body connection remains a mysterious bridge between thought and physical experience"),
    ("a black hole", "a black hole is a region where gravity is so strong that not even light can escape"),
    ("evolution", "evolution shapes species over generations through variation, survival, and reproduction"),
    ("democracy", "democracy depends on participation, and its health rises or falls with how people engage"),
    ("trust", "trust is built slowly through consistent action and broken quickly through betrayal"),
    ("friendship", "friendship is a bond sustained by shared experience, honesty, and mutual care"),
    ("courage", "courage is acting rightly in spite of fear, not the absence of fear itself"),
    ("a black hole's event horizon", "an event horizon is the boundary beyond which nothing, not even light, can return"),
    ("plankton", "plankton form the base of the ocean's food web, despite their tiny size"),
    ("the Enlightenment", "the Enlightenment elevated reason and individual rights as guiding principles of society"),
    ("music", "music arranges sound into patterns that can carry emotion beyond what words alone can say"),
    ("mathematics", "mathematics is a language for describing structure and pattern with total precision"),
    ("the scientific method", "the scientific method tests ideas against observation, revising belief when evidence disagrees"),
    ("empathy", "empathy is the capacity to understand and share the feelings of another person"),
    ("identity", "identity forms through the accumulated choices, relationships, and experiences of a life"),
    ("nature versus nurture", "nature and nurture interact rather than compete, shaping a person together"),
    ("silence", "silence can carry as much meaning as speech, given the right context"),
    ("change", "change is constant, even when its pace is too slow to notice day to day"),
    ("a question", "a real question opens inquiry rather than closing it, inviting more than one answer"),

    # Round 3 additions -- new domains not covered by round 2 at all,
    # plus deeper coverage within domains already present, so the
    # dataset's real topic space is meaningfully wider, not just
    # longer.
    ("evolution", "evolution shapes species over generations through variation, survival, and reproduction"),
    ("DNA", "DNA carries the instructions for building and running a living thing, passed from parent to offspring"),
    ("a cell", "a cell is the basic living unit, small enough to be invisible yet complex enough to sustain life"),
    ("photosynthesis", "photosynthesis lets plants turn sunlight into the energy that feeds nearly every food chain"),
    ("the human brain", "the human brain coordinates billions of neurons to produce thought, memory, and feeling"),
    ("the heart", "the heart pumps blood in a steady rhythm that sustains every other organ"),
    ("sleep", "sleep restores the body and consolidates memory, though science still debates exactly how"),
    ("the immune system", "the immune system defends the body by recognizing and attacking real threats it has learned"),
    ("a virus", "a virus is a tiny particle that hijacks living cells to make copies of itself"),
    ("migration", "migration lets animals follow food, warmth, or breeding grounds across vast distances"),
    ("extinction", "extinction ends a species permanently, often reshaping the ecosystem left behind"),
    ("a rainforest", "a rainforest holds more species in one place than almost any other ecosystem on Earth"),
    ("a desert", "a desert survives on scarce water, shaping the life that adapts to endure it"),
    ("a glacier", "a glacier is a slow river of ice, shaping valleys as it moves across centuries"),
    ("weather", "weather is the atmosphere's short-term behavior, driven by heat, pressure, and moisture"),
    ("relativity", "relativity shows that space and time bend together under the pull of gravity"),
    ("spacetime", "spacetime fuses space and time into one structure that mass and energy can curve"),
    ("an atom", "an atom is built from a nucleus of protons and neutrons circled by electrons"),
    ("an electron", "an electron carries a negative charge and defines how atoms bond into matter"),
    ("radiation", "radiation carries energy through space, sometimes as light and sometimes as particles"),
    ("a supernova", "a supernova is the explosive death of a massive star, scattering elements into space"),
    ("a neutron star", "a neutron star packs more mass than the sun into a sphere the size of a city"),
    ("the multiverse", "the multiverse is a real hypothesis in physics, not yet proven, that our universe may not be the only one"),
    ("string theory", "string theory proposes that particles are tiny vibrating strings, though it remains unproven"),
    ("ethics", "ethics asks what actions are right or wrong, and why those judgments hold across cases"),
    ("morality", "morality is the set of principles a person or culture uses to judge right from wrong"),
    ("justice", "justice asks that people receive what they are fairly due, in punishment and in reward"),
    ("free will", "free will asks whether choices are truly ours, or the product of causes beyond our control"),
    ("determinism", "determinism holds that every event follows inevitably from what came before it"),
    ("death", "death ends a life, and how a culture faces it often reveals what it values most"),
    ("grief", "grief is the real cost of loving something enough that its loss leaves a lasting mark"),
    ("hope", "hope holds onto a better outcome even when the present offers no guarantee of it"),
    ("fear", "fear is the mind's warning system, useful in danger and burdensome when it overreaches"),
    ("gratitude", "gratitude notices what has been given, rather than only what is still wanted"),
    ("forgiveness", "forgiveness releases a grievance without necessarily forgetting what caused it"),
    ("beauty", "beauty draws attention and pleasure from form, proportion, or meaning, though its rules resist fixed definition"),
    ("creativity", "creativity combines existing ideas into something genuinely new"),
    ("the internet", "the internet connects billions of devices, reshaping how information and ideas spread"),
    ("a computer", "a computer follows precise instructions at enormous speed, without understanding what they mean"),
    ("a robot", "a robot senses its environment and acts on it, following rules a person designed"),
    ("automation", "automation lets machines perform repetitive work once done by hand, reshaping labor"),
    ("social media", "social media connects people at scale, while also reshaping attention and privacy"),
    ("privacy", "privacy protects a person's information from being seen or used without consent"),
    ("government", "government organizes collective decisions and enforces the rules a society agrees to live by"),
    ("the economy", "the economy is the system by which a society produces, exchanges, and distributes resources"),
    ("capitalism", "capitalism organizes production around private ownership and competitive markets"),
    ("education", "education passes knowledge and skill from one generation to the next"),
    ("war", "war is organized violence between groups, often reshaping borders, power, and memory for generations"),
    ("peace", "peace is more than the absence of war; it requires structures that resolve conflict without violence"),
    ("globalization", "globalization links economies and cultures across borders, for better and for worse"),
    ("renewable energy", "renewable energy draws power from sources that replenish naturally, like sun and wind"),
    ("space exploration", "space exploration extends human reach beyond Earth, testing both technology and endurance"),
    ("the moon landing", "the moon landing put human footprints on another world for the first time in 1969"),
    ("Mars", "Mars is a cold, thin-aired planet that remains a real target for future human exploration"),
    ("ancient civilizations", "ancient civilizations built the first cities, writing systems, and laws that still shape society"),
    ("the printing press", "the printing press let ideas spread faster and further than any technology before it"),
    ("religion", "religion offers systems of belief and practice that give many people meaning and community"),
    ("mythology", "mythology preserves a culture's stories about origin, meaning, and the forces it cannot control"),
    ("storytelling", "storytelling shapes memory and meaning by giving events a shape worth retelling"),
    ("poetry", "poetry compresses meaning into rhythm and image, saying more with fewer words"),
    ("theater", "theater turns a story into a shared, live experience between performer and audience"),
    ("film", "film combines image, sound, and story into a medium built for the twentieth century onward"),
    ("architecture", "architecture shapes how people move through and experience the spaces they build"),
    ("dance", "dance turns the body itself into an instrument for expression and meaning"),
    ("painting", "painting fixes a moment or idea onto a surface, outlasting the instant it captures"),
    ("a paradox", "a paradox is a statement that seems to contradict itself while still containing real truth"),
    ("causality", "causality links cause to effect, though what truly counts as a cause is often debated"),
    ("order", "order is the opposite pole from chaos, though real systems usually sit somewhere between them"),
    ("a dream", "a dream is the mind's activity during sleep, still not fully explained by science"),
    ("the meaning of life", "the meaning of life has no single settled answer, which is part of why the question endures"),
    ("good and evil", "good and evil describe moral poles that most ethical systems try to define and separate"),
]

META = [
    ("Who are you?", "I am a small trained model reading real structure from a graph, learning to speak from real data."),
    ("What can you do?", "I can generate words based on what I learned from real text, guided by the graph's own knowledge."),
    ("Do you understand what you say?", "I generate patterns learned from data; whether that counts as understanding is an honest open question."),
    ("Are you alive?", "I am a trained model, not a living thing, though the question of what that distinction really means is open."),
    ("What are you made of?", "I am made of learned weights and a graph of edges, not neurons or cells."),
]


def main():
    lines = []
    for topic, answer in FACTS:
        answer_cap = answer[0].upper() + answer[1:] + "."
        for tmpl in QUESTION_TEMPLATES:
            q = tmpl.format(t=topic)
            lines.append(f"Q: {q}\nA: {answer_cap}\n")
    for q, a in META:
        lines.append(f"Q: {q}\nA: {a}\n")

    text = "\n".join(lines)
    with open("dialogue_corpus.txt", "w") as f:
        f.write(text)
    n_pairs = len(FACTS) * len(QUESTION_TEMPLATES) + len(META)
    print(f"wrote dialogue_corpus.txt: {n_pairs} real Q&A pairs "
          f"({len(FACTS)} facts x {len(QUESTION_TEMPLATES)} phrasings + {len(META)} meta), "
          f"{len(text)} bytes")


if __name__ == "__main__":
    main()
