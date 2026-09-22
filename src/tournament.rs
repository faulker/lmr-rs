//! Split a large `choice` question into smaller rounds so we stay out of Laya's `choice:11+`
//! calibration bucket. That bucket's published temperature is ~0.1, which turns a modest
//! logit gap into a 100% confident (and often wrong) winner.
//!
//! Nested names that share a `›` parent stay in the same group.

use serde_json::{json, Map, Value};

use crate::sequence::{QType, Question};

/// Largest flat question we will ask in one forward pass (`choice:6-10`).
pub const FLAT_MAX: usize = 10;
/// Target members per group when packing a larger set.
pub const GROUP_CAP: usize = 8;

/// Parent key for a nested name (`Investment › Fee` → `Investment`).
pub fn family_id(name: &str) -> &str {
    name.split('›')
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(name)
}

/// Pack `keys` into groups of at most `GROUP_CAP`, keeping `›` siblings together and
/// producing at most `FLAT_MAX` groups so the next question stays well calibrated.
pub fn pack_choice_groups(keys: &[String]) -> Vec<Vec<usize>> {
    pack_indices(&(0..keys.len()).collect::<Vec<_>>(), keys, FLAT_MAX)
}

/// Same as `pack_choice_groups`, but only the subset `idxs`. `max_flat` is the largest
/// question we will ask in one pass (from `model.tournament_after`).
pub fn pack_indices(idxs: &[usize], keys: &[String], max_flat: usize) -> Vec<Vec<usize>> {
    let max_flat = max_flat.max(2);
    let cap = GROUP_CAP.min(max_flat);
    if idxs.len() <= max_flat {
        return vec![idxs.to_vec()];
    }
    let families = families_in_order(idxs, keys);
    let mut groups = pack_families(families, cap);
    if groups.len() == 1 && idxs.len() > max_flat {
        groups = idxs.chunks(cap).map(|c| c.to_vec()).collect();
    }
    if groups.len() > max_flat {
        let per = groups.len().div_ceil(max_flat).max(2);
        groups = groups
            .chunks(per)
            .map(|chunk| chunk.iter().flatten().copied().collect())
            .collect();
    }
    groups
}

/// Families in first-seen order. Members of one parent stay together even if they
/// were not adjacent in `idxs`.
fn families_in_order(idxs: &[usize], keys: &[String]) -> Vec<Vec<usize>> {
    let mut order: Vec<String> = Vec::new();
    let mut members: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for &i in idxs {
        let fam = family_id(&keys[i]).to_string();
        if !members.contains_key(&fam) {
            order.push(fam.clone());
        }
        members.entry(fam).or_default().push(i);
    }
    order
        .into_iter()
        .map(|f| members.remove(&f).unwrap_or_default())
        .collect()
}

fn pack_families(families: Vec<Vec<usize>>, cap: usize) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut cur = Vec::new();
    for fam in families {
        if !cur.is_empty() && cur.len() + fam.len() > cap {
            groups.push(std::mem::take(&mut cur));
        }
        if fam.len() > cap && cur.is_empty() {
            groups.push(fam);
        } else {
            cur.extend(fam);
        }
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

/// A choice question whose criteria are only `idxs`.
pub fn subset_question(q: &Question, keys: &[String], idxs: &[usize]) -> Question {
    let orig = q.criteria.as_ref().and_then(Value::as_object);
    let mut map = Map::new();
    for &i in idxs {
        let value = orig
            .and_then(|m| m.get(&keys[i]))
            .cloned()
            .unwrap_or(Value::Null);
        map.insert(keys[i].clone(), value);
    }
    Question {
        qtype: QType::Choice,
        instructions: q.instructions.clone(),
        criteria: Some(Value::Object(map)),
    }
}

/// A choice question whose options are groups: the key lists the member names and the
/// description concatenates their criteria so the first pass still sees "grocery store".
pub fn groups_question(q: &Question, keys: &[String], groups: &[Vec<usize>]) -> Question {
    let orig = q.criteria.as_ref().and_then(Value::as_object);
    let mut map = Map::new();
    for group in groups {
        let label = group
            .iter()
            .map(|&i| keys[i].as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let mut parts = Vec::new();
        for &i in group {
            if let Some(Value::String(s)) = orig.and_then(|m| m.get(&keys[i])) {
                if !s.is_empty() {
                    parts.push(format!("{}: {s}", keys[i]));
                }
            }
        }
        let desc = if parts.is_empty() {
            Value::Null
        } else {
            json!(parts.join("; "))
        };
        map.insert(label, desc);
    }
    Question {
        qtype: QType::Choice,
        instructions: q.instructions.clone(),
        criteria: Some(Value::Object(map)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user_keys() -> Vec<String> {
        [
            "Alcohol",
            "Auto",
            "Bills",
            "Donation",
            "Entertainment",
            "Fee",
            "Food",
            "Gas",
            "Health",
            "House",
            "Investment",
            "Investment › Fee",
            "Investment › Interest",
            "Investment › Transaction",
            "Kids",
            "Legal",
            "Service",
            "Shopping",
            "Subscriptions",
            "Travel",
            "Utilities",
            "other",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }

    #[test]
    fn family_id_splits_on_nested_marker() {
        assert_eq!(family_id("Investment › Fee"), "Investment");
        assert_eq!(family_id("Investment"), "Investment");
        assert_eq!(family_id("Utilities › Electric"), "Utilities");
        assert_eq!(family_id("Food"), "Food");
    }

    #[test]
    fn twenty_two_categories_become_three_calibrated_groups() {
        let keys = user_keys();
        let groups = pack_choice_groups(&keys);
        assert!(
            groups.len() >= 2 && groups.len() <= FLAT_MAX,
            "groups {groups:?}"
        );
        assert!(
            groups.iter().all(|g| g.len() <= GROUP_CAP + 3),
            "{groups:?}"
        );
        let invest: Vec<usize> = (10..=13).collect();
        let home = groups
            .iter()
            .find(|g| g.contains(&10))
            .expect("investment group");
        for i in &invest {
            assert!(home.contains(i), "investment family split: {groups:?}");
        }
        let flat: Vec<usize> = groups.iter().flatten().copied().collect();
        let mut sorted = flat.clone();
        sorted.sort();
        assert_eq!(sorted, (0..22).collect::<Vec<_>>());
    }

    #[test]
    fn ten_or_fewer_stay_flat() {
        let keys: Vec<String> = (0..9).map(|i| format!("k{i}")).collect();
        assert_eq!(pack_choice_groups(&keys), vec![(0..9).collect::<Vec<_>>()]);
    }

    #[test]
    fn fifteen_flat_names_split_in_two() {
        let keys: Vec<String> = (0..15).map(|i| format!("Option {i}")).collect();
        let groups = pack_choice_groups(&keys);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups.iter().map(|g| g.len()).sum::<usize>(), 15);
    }

    #[test]
    fn subset_and_groups_keep_descriptions() {
        let q = Question::from_def(
            QType::Choice,
            &json!("Which?"),
            Some(&json!({
                "Food": "grocery store",
                "Shopping": "retail goods",
                "other": null
            })),
        )
        .unwrap();
        let keys = q.choice_keys();
        let sub = subset_question(&q, &keys, &[0, 2]);
        assert_eq!(sub.choice_keys(), vec!["Food", "other"]);
        let grouped = groups_question(&q, &keys, &[vec![0, 1], vec![2]]);
        let gkeys = grouped.choice_keys();
        assert_eq!(gkeys[0], "Food, Shopping");
        assert!(crate::sequence::render_options(&grouped)
            .unwrap()
            .join(" ")
            .contains("grocery store"));
    }
}
