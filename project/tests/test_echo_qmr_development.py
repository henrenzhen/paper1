import unittest

import numpy as np
import pandas as pd

from project.experiments.gsad.run_echo_qmr_development import select_prior_pool


class EchoQMRDevelopmentTests(unittest.TestCase):
    def test_pool_selection_chooses_helpful_external_prior(self):
        local = np.asarray([[0.6, 0.4], [0.6, 0.4]])
        prior = np.asarray([[0.1, 0.9], [0.1, 0.9]])
        validation = pd.DataFrame(
            {"target": ["B", "B"], "root": ["g1", "g2"]}
        )
        selected, pooled, weights = select_prior_pool(
            local_probabilities=local,
            prior_probabilities=prior,
            local_support=np.asarray([1.0, 1.0]),
            prior_available=np.asarray([True, True]),
            validation=validation,
            vocab=("A", "B"),
            strengths=(0.0, 1.0),
            kappas=(5.0,),
        )
        self.assertEqual(selected["strength"], 1.0)
        self.assertTrue((weights > 0).all())
        self.assertTrue((np.argmax(pooled, axis=1) == 1).all())


if __name__ == "__main__":
    unittest.main()
