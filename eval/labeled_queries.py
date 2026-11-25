"""Hand-labeled queries for Phase 2 retrieval evaluation.

Each entry pairs a natural-language query with the wiki ids of books that should
count as relevant. Labels are deliberately coarse: the harness measures whether
retrieval finds the right books, not whether it matches one canonical ranking.

Schema for every entry in `QUERIES`:

- `id`: stable slug used in result JSON and console reports. Changing it breaks
  historical comparisons, so treat it like a primary key.
- `query`: the exact text sent to `embed_query`. Phrasing is natural language on
  purpose; embedding the title verbatim would not test realistic search.
- `relevant_ids`: wiki ids that satisfy the query. They are strings because the
  corpus ids are numeric in parquet but `VectorStore` results are strings.
- `note`: human rationale for the labels, useful when revisiting a miss.

Most queries target one unambiguous classic. `cozy-village-mystery` is the
exception: it is the Phase 1 sanity query and labels the plausible mystery
cluster from its top results rather than a single canonical title.
"""

from typing import Any

# Add new queries here, never inside the harness. Every id is validated against
# books.parquet at startup so a stale label fails loudly before a run is scored.
QUERIES: list[dict[str, Any]] = [
    {
        "id": "cozy-village-mystery",
        "query": "a cozy mystery set in a small village",
        "relevant_ids": [
            "2074207",  # The Jungle Pyramid
            "14857283",  # The Mystery of the Cape Cod Tavern
            "6900207",  # The Mysterious Mannequin
            "9553753",  # The Kachina Doll Mystery
            "6899604",  # The Quest of the Missing Map
            "6914967",  # The Secret of the Forgotten City
            "21844678",  # Smaller and Smaller Circles
        ],
        "note": "Phase 1 sanity query; labels are the plausible mystery cluster from its top results.",
    },
    {
        "id": "totalitarian-farm-satire",
        "query": "a satirical novel about farm animals and totalitarian politics",
        "relevant_ids": ["620"],  # Animal Farm
        "note": "Title and summary both describe the pigs' takeover of the farm.",
    },
    {
        "id": "violent-dystopian-conditioning",
        "query": "a dark dystopian story about a violent teenager and behavior conditioning",
        "relevant_ids": ["843"],  # A Clockwork Orange
        "note": "Alex and the Ludovico Technique are the defining details.",
    },
    {
        "id": "hobbit-ring-quest",
        "query": "a hobbit's journey to reclaim a mountain kingdom with a wizard and a ring",
        "relevant_ids": ["30292"],  # The Hobbit
        "note": "Bilbo, Gandalf, and the ring are unique identifiers.",
    },
    {
        "id": "desert-fox-prince",
        "query": "a prince learns about love and loss in the desert from a fox",
        "relevant_ids": ["63425"],  # The Little Prince
        "note": "The fox and the desert are the book's best-known images.",
    },
    {
        "id": "spider-piglet-barn",
        "query": "a spider saves a piglet in a rural barn",
        "relevant_ids": ["302301"],  # Charlotte's Web
        "note": "Charlotte and Wilbur are the obvious targets.",
    },
    {
        "id": "walled-garden-renewal",
        "query": "a neglected girl finds friendship and renewal in a hidden garden",
        "relevant_ids": ["410873"],  # The Secret Garden
        "note": "Mary Lennox and the locked garden are distinctive.",
    },
    {
        "id": "civil-war-sisters",
        "query": "a family of sisters growing up during the American Civil War",
        "relevant_ids": ["8595912"],  # Little Women
        "note": "The March sisters are the target.",
    },
    {
        "id": "created-creature-gothic",
        "query": "a gothic novel about a scientist who creates and abandons a living creature",
        "relevant_ids": ["18580673"],  # Frankenstein
        "note": "Use the original novel id, not later adaptations.",
    },
    {
        "id": "vampire-count-england",
        "query": "a Victorian vampire count who moves to England and hunts a young woman",
        "relevant_ids": ["7923"],  # Dracula
        "note": "Count Dracula is the canonical target.",
    },
    {
        "id": "holmes-moors-manor",
        "query": "Sherlock Holmes investigates a cursed manor on the moors",
        "relevant_ids": ["30280"],  # The Hound of the Baskervilles
        "note": "The moor setting separates it from other Holmes stories.",
    },
    {
        "id": "pirates-buried-map",
        "query": "a boy's adventure with pirates, a buried map, and a one-legged sailor",
        "relevant_ids": ["167963"],  # Treasure Island
        "note": "Jim Hawkins and Long John Silver are the targets.",
    },
    {
        "id": "around-world-eighty-days",
        "query": "a wager to travel around the world in eighty days by train and steamer",
        "relevant_ids": ["42609"],  # Around the World in Eighty Days
        "note": "Phileas Fogg's wager is unique.",
    },
    {
        "id": "time-travel-far-future",
        "query": "a machine that lets a traveller visit the far future",
        "relevant_ids": ["29834"],  # The Time Machine
        "note": "H. G. Wells's Time Traveller is the target.",
    },
    {
        "id": "submarine-giant-squid",
        "query": "a captain explores the ocean in a submarine and fights a giant squid",
        "relevant_ids": ["38093"],  # Twenty Thousand Leagues Under the Sea
        "note": "Captain Nemo and the Nautilus are the targets.",
    },
    {
        "id": "children-castle-adventure",
        "query": "children uncover a mystery in a castle during a summer adventure",
        "relevant_ids": ["23943721"],  # The Castle of Adventure
        "note": "Enid Blyton's adventure story is the target.",
    },
    {
        "id": "old-fisherman-marlin",
        "query": "a sparse novella about an old fisherman and a giant marlin",
        "relevant_ids": ["161502"],  # The Old Man and the Sea
        "note": "Santiago and the marlin are the targets.",
    },
    {
        "id": "war-satire-yossarian",
        "query": "a satirical war novel about an airman named Yossarian",
        "relevant_ids": ["55469"],  # Catch-22
        "note": "Yossarian is an unambiguous identifier.",
    },
    {
        "id": "magical-realism-macondo",
        "query": "a magical-realism family saga in the town of Macondo",
        "relevant_ids": ["654705"],  # One Hundred Years of Solitude
        "note": "Macondo and the Buendía family are the targets.",
    },
    {
        "id": "wardrobe-winter-world",
        "query": "a lion, a witch, and a wardrobe open a portal to a winter world",
        "relevant_ids": ["242130"],  # The Lion, the Witch and the Wardrobe
        "note": "Aslan, the White Witch, and the wardrobe are the targets.",
    },
]
