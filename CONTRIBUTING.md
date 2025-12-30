## How to Add A Missing Algorithm / Language

### Step 1: Add the algorithm to the `ALGORITHMS.md`

Modify [`ALGORITHMS.md`](https://github.com/codereport/hoogle-translate/blob/main/ALGORITHMS.md) by adding a row with the relevant information. For example:

| Language | Algorithm Name | AlgoId |     Lib     | Expr  |                           Doc Link                            |
| :------: | :------------: | :----: | :---------: | :---: | :-----------------------------------------------------------: |
|   C++    |  `accumulate`  |   1    | `<numeric>` |       | [doc](https://en.cppreference.com/w/cpp/algorithm/accumulate) |

Information to be included:

1. **Language**
2. **Algorithm Name**
3. **AlgoId:** this is a unique id that links algorithms to a category - check to see if it already exists
4. **Lib:** this is the name of the library that the algorithm belongs to
5. **Expr:** mark as `Y` if this is not a single function but an expression or combination of functions
6. **Doc Link:** this is the link to the documentation of the algorithm

### Step 2: Check if programming language logo exists

Check that the language logo for the image is in `public/media/logos/`. If it isn't please add one, and update the `src/imgs.cljs` accordingly.
