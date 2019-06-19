import os
import pytest
import sys

sys.path.append('..')

from utils.Gemima_Utils import imgLoader


def listLoad():
    # Test file format: UCLH_01493856_1_0001_L

    TEST_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/scripts/test/"
    HI_PATH = TEST_PATH + "test_vols/Hi/"
    LO_PATH = TEST_PATH + "test_vols/Lo/"

    hi_list = os.listdir(HI_PATH)
    lo_list = os.listdir(LO_PATH)

    hi_list = list(map(lambda img: HI_PATH + img, hi_list))
    lo_list = list(map(lambda img: LO_PATH + img, lo_list))

    return hi_list, lo_list


def test_imgLoader_pairs():
    hi_list, lo_list = listLoad()
    indices = [0, 1, 2, 3, 4, 5, 6, 7]

    test_hi_mb = [hi_list[i][-26:-4] for i in indices]
    test_lo_mb = [lo_list[i][-26:-4] for i in indices]

    # Check that minibatch lengths are equal
    assert len(test_hi_mb) == len(test_lo_mb)
    # Check that subject IDs match
    assert [vol[:-1] for vol in test_hi_mb] == [vol[:-1] for vol in test_lo_mb]
    # Check that images are paired appropriately
    assert 'L' not in [vol[-1:] for vol in test_hi_mb]
    assert 'H' not in [vol[-1:] for vol in test_lo_mb]


def test_neg_imgLoader_pairs():
    hi_list, lo_list = listLoad()

    # Check error thrown if index exceeds list length
    with pytest.raises(IndexError):
        indices = [0, 1, 2, 19]
        _, _ = imgLoader(hi_list, lo_list, indices)

    # Check error thrown if minibatch lengths are unequal
    with pytest.raises(AssertionError):
        indices = [0, 1, 2, 3, 4, 5, 6, 7]
        test_hi_mb = [hi_list[i][-26:-4] for i in indices]
        indices = [0, 1, 2, 3, 4, 5, 6]
        test_lo_mb = [lo_list[i][-26:-4] for i in indices]
        assert len(test_hi_mb) == len(test_lo_mb)

    # Check error thrown if subject IDs do not match
    with pytest.raises(AssertionError):
        indices = [0, 1, 2, 3]
        test_hi_mb = [hi_list[i][-26:-4] for i in indices]
        indices = [4, 5, 6, 7]
        test_lo_mb = [lo_list[i][-26:-4] for i in indices]
        assert [vol[:-1] for vol in test_hi_mb] == [vol[:-1] for vol in test_lo_mb]

    # Check error thrown if images are not paired appropriately
    with pytest.raises(AssertionError):
        indices = [0, 1, -2, 3, 4, 5, 6, 7]
        test_hi_mb = [hi_list[i][-26:-4] for i in indices]
        indices = [0, 1, 2, 3, 4, -5, 6, 7]
        test_lo_mb = [lo_list[i][-26:-4] for i in indices]

        if any(i < 0 for i in indices):
            test_hi_mb = [lo_list[i][-26:-4] for i in indices if i < 0]
            test_lo_mb = [hi_list[i][-26:-4] for i in indices if i < 0]
            assert 'L' not in [vol[-1:] for vol in test_hi_mb]
            assert 'H' not in [vol[-1:] for vol in test_lo_mb]


def test_imgLoader_dims():
    TENSOR_DIMS = 5
    hi_list, lo_list = listLoad()
    indices = [0, 1, 2, 3, 4, 5, 6, 7]

    test_hi_mb = [hi_list[i][-26:-4] for i in indices]
    test_lo_mb = [lo_list[i][-26:-4] for i in indices]
    
    hi_img, lo_img = imgLoader(hi_list, lo_list, indices)
    
    # Check that minibatch tensor shapes are equal
    assert hi_img.shape == lo_img.shape
    # Check that tensor shapes are correct
    assert len(hi_img.shape) == TENSOR_DIMS


def test_neg_imgLoader_dims():
    hi_list, lo_list = listLoad()
    indices = [0, 1, 2, 3, 4, 5, 6, 7]

    test_hi_mb = [hi_list[i][-26:-4] for i in indices]
    test_lo_mb = [lo_list[i][-26:-4] for i in indices]
    
    hi_img, lo_img = imgLoader(hi_list, lo_list, indices)
    
    # Check that minibatch tensor shapes are equal
    assert hi_img.shape == lo_img.shape
    # Check that tensor shapes are correct
    assert len(hi_img.shape) == TENSOR_DIMS

