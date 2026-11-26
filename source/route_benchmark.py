"""Measured results from the 1–24 product, five-board benchmark (120 layout groups)."""

from typing import NamedTuple

from planner import FULL_AUTO


class Result(NamedTuple):
    median_elapsed_s: float
    penalized_total_s: float
    passed_boards: int

    @property
    def boards_per_minute(self):
        return 5 * 60 / self.median_elapsed_s


# Each row is the median elapsed time for five boards, the total scored time
# (20 s per failed board), and the number of passing boards out of 600.
# Source: docs/트랙2_연속5판_1-24개_전략비교.md. Rank using unrounded measurements.
AUTOMATIC_RESULT = Result(68.97863266763801, 8509.064622425261, 256)
FIXED_RESULTS = {
    "home": {
        "horizontal": Result(69.86387760026044, 10400.510201758116, 108),
        "direct": Result(71.69967098396316, 10528.105303439737, 100),
        "oblique": Result(70.33022567249836, 10410.933428039882, 106),
        "order": Result(73.86203378834253, 10510.384375921702, 103),
        "target": Result(78.3154490498324, 11053.227395468839, 64),
        "raster": Result(95.90557480286543, 12000, 0),
        "leading": Result(73.49518600095668, 9331.66588760157, 208),
    },
    "point": {
        "horizontal": Result(69.88009778048752, 10503.850374860514, 99),
        "direct": Result(71.6762623922779, 10520.510983083359, 100),
        "oblique": Result(70.34398098240186, 10458.927698161879, 101),
        "order": Result(73.83747092458273, 10587.015916049608, 96),
        "target": Result(78.29191550207273, 11089.662181532902, 60),
        "raster": Result(96.09862789736538, 12000, 0),
        "leading": Result(73.54352500824847, 9357.314125964474, 207),
    },
    "continue": {
        "horizontal": Result(68.9671022716953, 10610.788906622949, 86),
        "direct": Result(71.27980633535239, 10570.132881807953, 92),
        "oblique": Result(69.79555660142933, 10587.974452626211, 90),
        "order": Result(73.9128364661876, 10661.237073843611, 87),
        "target": Result(77.8257273691639, 11081.828455474079, 57),
        "raster": Result(91.82169771565589, 12000, 0),
        "leading": Result(72.72743762570914, 9315.923402850998, 204),
    },
    "top": {
        "horizontal": Result(69.03927193359732, 11153.738368043296, 49),
        "direct": Result(71.31642696876679, 11102.698617487873, 54),
        "oblique": Result(69.6619141224718, 11137.900920556742, 51),
        "order": Result(73.85901632460684, 11098.901321360792, 57),
        "target": Result(77.18906031529701, 11181.557197330369, 51),
        "raster": Result(91.82169771565589, 12000, 0),
        "leading": Result(72.9798072789294, 10451.096426957954, 114),
    },
    "bottom": {
        "horizontal": Result(68.79280931344161, 11052.453796793267, 58),
        "direct": Result(71.13614123723082, 11029.694456223931, 61),
        "oblique": Result(69.77854232971245, 11067.342138582275, 59),
        "order": Result(73.77291377299036, 10947.595447606966, 67),
        "target": Result(77.84931081584605, 11202.928373710156, 50),
        "raster": Result(91.82169771565589, 12000, 0),
        "leading": Result(72.65388196682147, 10152.568449866261, 137),
    },
    "leading_top": {
        "horizontal": Result(68.71355885075681, 10746.642567850933, 77),
        "direct": Result(70.42222562848045, 10754.28943778608, 79),
        "oblique": Result(69.11885991675436, 10606.590872773006, 88),
        "order": Result(72.67727445769111, 10601.40844245276, 93),
        "target": Result(76.3616188388965, 11177.534208740663, 50),
        "raster": Result(91.82169771565589, 12000, 0),
        "leading": Result(72.70331161839388, 10093.570081449525, 143),
    },
}
# All 600 layouts had a unique leading X. Both tie-breakers produced identical runs.
FIXED_RESULTS["leading_bottom"] = FIXED_RESULTS["leading_top"]


def ranked_routes(motion):
    fixed = FIXED_RESULTS[motion.advance or motion.action]
    # All candidates share the same full-rack baseline for this setting.
    # A lower scored total therefore gives a greater percentage saving.
    ranked = sorted(fixed, key=lambda route: fixed[route].penalized_total_s)
    return {
        FULL_AUTO: AUTOMATIC_RESULT,
        **{mode: fixed[mode] for mode in ranked},
    }
