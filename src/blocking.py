import re
import pandas as pd
from collections import defaultdict, Counter


STRONG_VIEWS = {"strong_name"}
RARE_TOKEN_VIEW = "rare_token"


# ============================================================
# BASIC TEXT UTILITIES
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_address_numerics(address):
    if pd.isna(address):
        return set()

    return set(re.findall(r'\b\d+\b', str(address)))


def tokenize(text):
    norm = normalize_text(text)

    if not norm:
        return []

    return norm.split()


# ============================================================
# PHASE 0 PREPROCESSOR
# ============================================================

class Phase0Preprocessor:

    def __init__(self):
        self.token_df = Counter()
        self.name_df = Counter()
        self.total_docs = 0

    def fit(self, corpus_df):

        self.total_docs = len(corpus_df)

        name_col = (
            'name'
            if 'name' in corpus_df.columns
            else corpus_df.columns[1]
        )

        names = corpus_df[name_col].fillna('').astype(str)

        print("  Computing normalized names...")

        normalized_names = (
            names.str.lower()
                 .str.replace(
                     r'[^a-z0-9\s]',
                     ' ',
                     regex=True
                 )
                 .str.replace(
                     r'\s+',
                     ' ',
                     regex=True
                 )
                 .str.strip()
        )

        print("  Computing name frequencies...")

        name_counts = normalized_names[
            normalized_names != ''
        ].value_counts()

        self.name_df.update(
            name_counts.to_dict()
        )

        print("  Computing token frequencies...")

        # Token DF: each token counted once per business name
        token_series = normalized_names[
            normalized_names != ''
        ].map(
            lambda x: set(x.split())
        )

        token_counts = Counter()

        for tokens in token_series:
            token_counts.update(tokens)

        self.token_df = token_counts

        return self

    def transform_record(self, row, id_col='id'):

        rec_id = row.get(id_col, '')

        if pd.notna(rec_id):
            rec_id = str(rec_id)
        else:
            rec_id = ""

        name = row.get('name', '')
        address = row.get('address', '')

        norm_name = normalize_text(name)

        tokens = tokenize(name)

        prefix = (
            norm_name[:4]
            if len(norm_name) >= 4
            else norm_name
        )

        char_ngrams = set()

        if len(norm_name) >= 3:

            for i in range(len(norm_name) - 2):
                char_ngrams.add(
                    norm_name[i:i + 3]
                )

        addr_nums = extract_address_numerics(address)

        address_exists = (
            pd.notna(address)
            and bool(str(address).strip())
        )

        return {
            'id': rec_id,
            'norm_name': norm_name,
            'tokens': tokens,
            'prefix': prefix,
            'char_ngrams': char_ngrams,
            'addr_nums': addr_nums,
            'address_exists': address_exists
        }


# ============================================================
# BLOCKING CONFIG
# ============================================================

class BlockingConfig:

    def __init__(
        self,
        strong_name_df_threshold=50,
        rare_token_df_threshold=50,
        k=2,
        rare_token_mode="precision",
        max_candidates_per_s1=None
    ):

        self.strong_name_df_threshold = (
            strong_name_df_threshold
        )

        self.rare_token_df_threshold = (
            rare_token_df_threshold
        )

        self.k = k

        self.rare_token_mode = (
            rare_token_mode
        )

        self.max_candidates_per_s1 = (
            max_candidates_per_s1
        )

        self.train_token_df = None
        self.train_name_df = None

    def freeze_train_stats(self, preprocessor):

        self.train_token_df = (
            preprocessor.token_df
        )

        self.train_name_df = (
            preprocessor.name_df
        )


# ============================================================
# PHASE 1 BLOCKER
# ============================================================

class Phase1Blocker:

    def __init__(self, config):

        self.config = config

        self.inverted_index = defaultdict(set)


    # --------------------------------------------------------
    # OPTIMIZED DATAFRAME INDEX BUILD
    # --------------------------------------------------------

    def build_index_from_df(
        self,
        corpus_df,
        chunk_size=250_000
    ):

        total = len(corpus_df)

        print(
            f"  Total corpus records: {total:,}"
        )

        for start in range(
            0,
            total,
            chunk_size
        ):

            end = min(
                start + chunk_size,
                total
            )

            chunk = corpus_df.iloc[
                start:end
            ]

            print(
                f"  Processing {start:,} - {end:,}..."
            )

            for row in chunk.itertuples(
                index=False
            ):

                rid = str(row.id)

                name = normalize_text(
                    row.name
                )

                address = (
                    ""
                    if pd.isna(row.address)
                    else str(row.address)
                )

                # --------------------------------------------
                # 1. EXACT NORMALIZED NAME
                # --------------------------------------------

                if name:

                    self.inverted_index[
                        ('norm_name', name)
                    ].add(rid)


                # --------------------------------------------
                # 2. RARE TOKEN
                # --------------------------------------------

                if name:

                    tokens = set(
                        name.split()
                    )

                    for token in tokens:

                        df = (
                            self.config
                            .train_token_df
                            .get(
                                token,
                                999999
                            )
                        )

                        if (
                            df
                            <= self.config
                            .rare_token_df_threshold
                        ):

                            self.inverted_index[
                                ('token', token)
                            ].add(rid)


                # --------------------------------------------
                # 3. PREFIX
                # --------------------------------------------

                if name:

                    prefix = name[:4]

                    self.inverted_index[
                        ('prefix', prefix)
                    ].add(rid)


                # --------------------------------------------
                # 4. CHARACTER TRIGRAMS
                # --------------------------------------------

                if len(name) >= 3:

                    ngrams = {
                        name[i:i + 3]
                        for i in range(
                            len(name) - 2
                        )
                    }

                    for ng in ngrams:

                        self.inverted_index[
                            ('ngram', ng)
                        ].add(rid)


                # --------------------------------------------
                # 5. ADDRESS NUMBERS
                # --------------------------------------------

                if address:

                    nums = set(
                        re.findall(
                            r'\b\d+\b',
                            address
                        )
                    )

                    for num in nums:

                        self.inverted_index[
                            ('addr_num', num)
                        ].add(rid)


            print(
                f"  Indexed {end:,}/{total:,} "
                f"({end / total * 100:.1f}%)",
                flush=True
            )


    # --------------------------------------------------------
    # RAW CANDIDATE RETRIEVAL
    # --------------------------------------------------------

    def get_raw_candidate_views(
        self,
        s1_rec
    ):

        candidate_views = defaultdict(set)


        # Exact normalized name
        if s1_rec['norm_name']:

            for cid in self.inverted_index[
                (
                    'norm_name',
                    s1_rec['norm_name']
                )
            ]:

                candidate_views[
                    str(cid)
                ].add(
                    (
                        'norm_name_val',
                        s1_rec['norm_name']
                    )
                )


        # Rare tokens
        for token in set(
            s1_rec['tokens']
        ):

            df = (
                self.config
                .train_token_df
                .get(
                    token,
                    999999
                )
            )

            if (
                df
                <= self.config
                .rare_token_df_threshold
            ):

                for cid in self.inverted_index[
                    ('token', token)
                ]:

                    candidate_views[
                        str(cid)
                    ].add(
                        RARE_TOKEN_VIEW
                    )


        # Prefix
        if s1_rec['prefix']:

            for cid in self.inverted_index[
                (
                    'prefix',
                    s1_rec['prefix']
                )
            ]:

                candidate_views[
                    str(cid)
                ].add(
                    "prefix"
                )


        # Character ngrams
        for ng in s1_rec[
            'char_ngrams'
        ]:

            for cid in self.inverted_index[
                ('ngram', ng)
            ]:

                candidate_views[
                    str(cid)
                ].add(
                    "ngram"
                )


        # Address numeric
        for num in s1_rec[
            'addr_nums'
        ]:

            for cid in self.inverted_index[
                ('addr_num', num)
            ]:

                candidate_views[
                    str(cid)
                ].add(
                    "address_numeric"
                )


        return candidate_views


    # --------------------------------------------------------
    # RAW UNION
    # --------------------------------------------------------

    def get_union_candidates(
        self,
        raw_candidate_views
    ):

        return set(
            str(cid)
            for cid in raw_candidate_views.keys()
        )


    # --------------------------------------------------------
    # PRECISION-AWARE CANDIDATE FILTER
    # --------------------------------------------------------

    def evaluate_candidates(
        self,
        raw_candidate_views,
        s1_rec
    ):

        admitted_candidates = set()


        for cid, raw_views in (
            raw_candidate_views.items()
        ):

            cid_str = str(cid)

            views = set()


            # Convert raw signals
            for v in raw_views:

                if (
                    isinstance(v, tuple)
                    and v[0]
                    == 'norm_name_val'
                ):

                    n_val = v[1]

                    df = (
                        self.config
                        .train_name_df
                        .get(
                            n_val,
                            1
                        )
                    )

                    if (
                        df
                        <= self.config
                        .strong_name_df_threshold
                    ):

                        views.add(
                            "strong_name"
                        )

                    else:

                        views.add(
                            "weak_name"
                        )

                else:

                    views.add(v)


            # Strong exact name
            if "strong_name" in views:

                admitted_candidates.add(
                    cid_str
                )

                continue


            admitted = False


            # --------------------------------------------
            # Rare token route
            # --------------------------------------------

            if (
                self.config
                .rare_token_mode
                == "diagnostic"
            ):

                if (
                    RARE_TOKEN_VIEW
                    in views
                ):

                    admitted = True

            else:

                if (
                    RARE_TOKEN_VIEW
                    in views
                ):

                    if s1_rec[
                        'address_exists'
                    ]:

                        if (
                            "address_numeric"
                            in views
                        ):

                            admitted = True

                    else:

                        name_signals = (
                            views.intersection(
                                {
                                    "weak_name",
                                    "prefix",
                                    "ngram"
                                }
                            )
                        )

                        if (
                            len(name_signals)
                            >= 2
                        ):

                            admitted = True


            # --------------------------------------------
            # Multi-signal route
            # --------------------------------------------

            if not admitted:

                weak_signals = (
                    views.intersection(
                        {
                            "weak_name",
                            "prefix",
                            "ngram",
                            "address_numeric"
                        }
                    )
                )

                if (
                    len(weak_signals)
                    >= self.config.k
                ):

                    admitted = True


            if admitted:

                admitted_candidates.add(
                    cid_str
                )


        # Optional candidate cap
        if (
            self.config
            .max_candidates_per_s1
            and len(admitted_candidates)
            > self.config
            .max_candidates_per_s1
        ):

            admitted_candidates = set(
                sorted(
                    admitted_candidates
                )[
                    :self.config
                    .max_candidates_per_s1
                ]
            )


        return admitted_candidates