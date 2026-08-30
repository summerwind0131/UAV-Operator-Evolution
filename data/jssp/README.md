# JSSP datasets

`orlib/jobshop1.txt` is an unmodified archive downloaded from the official
OR-Library endpoint. `orlib/legal.html` is the official legal page saved on the
same retrieval date. Their URLs and SHA-256 digests are recorded in
`orlib/SOURCE.json`.

The 60 training instances are generated deterministically by
`jssp_operator_evolution.data.generate_training_instances` with master seed
`20260823`. The 82 classic instances are normalized, sorted by
`(source_family, jobs, machines, content_hash)`, then assigned by zero-based
position: even positions to validation and odd positions to test. Test
instances and the test manifest are available through `JSSPDatasetSplits` only
after that object has issued a population-freeze receipt.
