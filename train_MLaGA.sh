DATASET_MAPPING="Movies:nc,Arts:nc,VideoGames:nc,RedditS:nc,Health:lp,Beauty:lp,CD:lp,Art500K:lp"

./scripts/train.sh \
    vicuna_2layer \
    nc-lp \
    Movies-Arts-VideoGames-RedditS-Health-Beauty-CD-Art500K \
    8 \
    clip_mix \
    TIQ_demo \
    4 \
    True \
    True \
    1 \
    Movies:nc,Arts:nc,VideoGames:nc,RedditS:nc,Health:lp,Beauty:lp,CD:lp,Art500K:lp
