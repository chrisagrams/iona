# Changelog

## [0.1.1](https://github.com/chrisagrams/iona/compare/v0.1.0...v0.1.1) (2026-10-07)


### Features

* add train_fraction for pretraining data scaling ([#44](https://github.com/chrisagrams/iona/issues/44)) ([6ab534e](https://github.com/chrisagrams/iona/commit/6ab534ede984202b4e701ff5b3ba6db4f136f9c7))


### Bug Fixes

* run mzML denoising through torch.compile ([#52](https://github.com/chrisagrams/iona/issues/52)) ([d295c55](https://github.com/chrisagrams/iona/commit/d295c55584388a4f5de3c638e8f097b8b451383f))
* support a portable telegraf binary on aurora ([#47](https://github.com/chrisagrams/iona/issues/47)) ([339bcae](https://github.com/chrisagrams/iona/commit/339bcaeb966ddc54628da57915c5e3f5524f7970))
* support python 3.12 and torch.compile on aurora ([#42](https://github.com/chrisagrams/iona/issues/42)) ([a47ff0f](https://github.com/chrisagrams/iona/commit/a47ff0f2272297cb00df398cbb834a2aea871d90))


### Performance Improvements

* apply attention padding mask once across layers ([#45](https://github.com/chrisagrams/iona/issues/45)) ([1530094](https://github.com/chrisagrams/iona/commit/1530094afb9aa00f09d730f6109df19a8702a49c))
* length-grouped batching for training ([#48](https://github.com/chrisagrams/iona/issues/48)) ([25008ff](https://github.com/chrisagrams/iona/commit/25008ffd8ede345340e64f334604ede7ef4500a4))
* stack delta bias head MLPs into one GEMM ([#46](https://github.com/chrisagrams/iona/issues/46)) ([3e4c467](https://github.com/chrisagrams/iona/commit/3e4c467978fbe0c4974066a6497b0c9faa272746))
