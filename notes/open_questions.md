# Open questions

Your running answer sheet. Answer in your own words, short is fine, and tick the box when done.
Add new questions as they come up during the build. Interviewers ask these.

## Data assessment (`notes/data_assessment.md`, sections E to G)

- [ ] **Missing vs. not billed.** Why is "no bill yet" a different problem from "data got lost"? How does that change the cleaning choice?
  > Your answer:
- [ ] **Duplicates.** The 22 rows that are identical apart from `customerID`: when would you drop them, and when would you keep them?
  > Your answer:
- [ ] **E. Fit for each module.** For each of the 4 modules, what does this CSV support and what doesn't it? (Which modules need text, time or conversation data?)
  > M1 analytics:
  > M2 churn model:
  > M3 agents / RAG:
  > M4 evals / drift:
- [ ] **F. Limitations vs. a real telco.** Three limitations, and what real data would add for each. What does a single snapshot rule out for drift monitoring?
  > Your answer:
- [ ] **G. License.** What does the dataset's Kaggle page say? If nothing, what goes in the README and what does that mean for a public repo?
  > Your answer:
- [ ] **xlsx decision.** We dropped the IBM xlsx. What did we give up (`Churn Reason`, `CLTV`), and what is the fallback for each?
  > Your answer:

## Cleaning step

- [ ] **Why 0 for the 11 blanks?** I filled `TotalCharges` with 0 for tenure-0 customers instead of dropping them or using the median. Argue for or against.
  > Your answer:
- [ ] **Why raise on other blanks?** `clean()` raises an error if a blank `TotalCharges` appears for a customer with tenure above 0. What failure does that protect against, and why is it better than silently filling in?
  > Your answer:
- [ ] **Why don't the tests check `isna()`?** Which test would have caught the "`isna()` says 0" trap, and why?
  > Your answer:
- [ ] **Where does the split happen?** `clean()` is applied before the train/test split. Is that leakage? Which steps are safe before the split and which are not (imputing with the mean, scaling, encoding)?
  > Your answer:
