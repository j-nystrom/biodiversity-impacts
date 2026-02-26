suppressPackageStartupMessages({
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Missing R package 'arrow'. Install with install.packages('arrow').")
  }
  if (!requireNamespace("glmmTMB", quietly = TRUE)) {
    stop("Missing R package 'glmmTMB'. Install with install.packages('glmmTMB').")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Missing R package 'jsonlite'. Install with install.packages('jsonlite').")
  }
  library(arrow)
  library(glmmTMB)
  library(jsonlite)
})

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (startsWith(arg, "--")) {
      kv <- sub("^--", "", arg)
      parts <- strsplit(kv, "=", fixed = TRUE)[[1]]
      key <- parts[1]
      value <- if (length(parts) > 1) {
        paste(parts[-1], collapse = "=")
      } else {
        ""
      }
      out[[key]] <- value
    }
  }
  out
}

get_family <- function(family, link) {
  if (family == "beta") {
    return(beta_family(link = link))
  }
  if (family == "gaussian") {
    return(gaussian(link = link))
  }
  stop(paste("Unsupported family:", family))
}

factorize_columns <- function(df, columns) {
  for (col in columns) {
    if (col %in% names(df)) {
      df[[col]] <- as.factor(df[[col]])
    }
  }
  df
}

align_factor_levels <- function(new_df, model, columns) {
  mf <- tryCatch(model.frame(model), error = function(e) NULL)
  if (is.null(mf)) return(new_df)

  for (col in columns) {
    if (col %in% names(new_df) && col %in% names(mf)) {
      new_df[[col]] <- factor(new_df[[col]], levels = levels(mf[[col]]))
    } else if (col %in% names(new_df)) {
      new_df[[col]] <- as.factor(new_df[[col]])
    }
  }
  new_df
}

args <- commandArgs(trailingOnly = TRUE)
params <- parse_args(args)

mode <- params[["mode"]]
training_data_path <- params[["training-data-path"]]
prediction_data_path <- params[["prediction-data-path"]]
model_path <- params[["model-path"]]
prediction_output_path <- params[["prediction-output-path"]]
effects_output_path <- params[["effects-output-path"]]
phi_output_path <- params[["phi-output-path"]]
formula_str <- params[["formula"]]
family <- params[["family"]]
link <- params[["link"]]
categorical_vars <- params[["categorical-vars"]]
ci_method <- params[["ci-method"]]
n_boot <- params[["n-boot"]]
seed <- params[["seed"]]
re_lower_perc <- params[["re-lower-perc"]]
re_upper_perc <- params[["re-upper-perc"]]

if (is.null(mode) || is.null(model_path)) {
  stop("Required args: --mode, --model-path")
}

if (is.null(family) || family == "") {
  family <- "beta"
}
if (is.null(link) || link == "") {
  link <- "logit"
}

message("glmmTMB runner mode: ", mode)
message("family/link: ", family, "/", link)

if (!is.null(seed) && nzchar(seed)) {
  set.seed(as.integer(seed))
}

cat_vars <- character(0)
if (!is.null(categorical_vars) && nzchar(categorical_vars)) {
  cat_vars <- strsplit(categorical_vars, ",")[[1]]
}

if (mode == "fit") {
  if (is.null(formula_str) || formula_str == "") {
    stop("Missing --formula for fit mode.")
  }
  if (is.null(training_data_path) || training_data_path == "") {
    stop("Missing --training-data-path for fit mode.")
  }
  message("Fitting model with formula: ", formula_str)

  df <- as.data.frame(read_parquet(training_data_path))
  df <- factorize_columns(df, cat_vars)
  model_formula <- as.formula(formula_str)
  model_family <- get_family(family, link)
  model <- glmmTMB(model_formula, data = df, family = model_family)
  saveRDS(model, model_path)
  quit(save = "no")
}

if (mode == "predict") {
  if (is.null(prediction_data_path) || prediction_data_path == "") {
    stop("Missing --prediction-data-path for predict mode.")
  }
  if (is.null(prediction_output_path) || prediction_output_path == "") {
    stop("Missing --prediction-output-path for predict mode.")
  }
  message("Making predictions...")

  df <- as.data.frame(read_parquet(prediction_data_path))
  model <- readRDS(model_path)
  df <- align_factor_levels(df, model, cat_vars)
  pred_fe <- predict(
    model,
    newdata = df,
    type = "response",
    re.form = NA,
    allow.new.levels = TRUE
  )
  pred_re <- predict(
    model,
    newdata = df,
    type = "response",
    re.form = NULL,
    allow.new.levels = TRUE
  )

  fixef_vals <- fixef(model)$cond
  intercept <- if ("(Intercept)" %in% names(fixef_vals)) {
    fixef_vals[["(Intercept)"]]
  } else {
    fixef_vals[[1]]
  }
  linkinv_func <- model$family$linkinv
  if (!is.function(linkinv_func)) {
    linkinv_func <- make.link(link)$linkinv
  }
  ref_val <- linkinv_func(intercept)
  ref_pred <- rep(ref_val, nrow(df))

  out <- data.frame(
    SSBS = as.character(df$SSBS),
    Predicted_RE = as.numeric(pred_re),
    Predicted_FE = as.numeric(pred_fe),
    Reference_pred_FE = as.numeric(ref_pred)
  )
  write_parquet(out, prediction_output_path)
  quit(save = "no")
}

if (mode == "extract-effects") {
  if (is.null(effects_output_path) || effects_output_path == "") {
    stop("Missing --effects-output-path for extract-effects mode.")
  }
  model <- readRDS(model_path)

  ci_method <- tolower(ifelse(is.null(ci_method) || ci_method == "", "wald", ci_method))
  n_boot <- as.integer(ifelse(is.null(n_boot) || n_boot == "", "200", n_boot))
  seed <- as.integer(ifelse(is.null(seed) || seed == "", "42", seed))
  re_lower <- as.numeric(
    ifelse(is.null(re_lower_perc) || re_lower_perc == "", "5", re_lower_perc)
  )
  re_upper <- as.numeric(
    ifelse(is.null(re_upper_perc) || re_upper_perc == "", "95", re_upper_perc)
  )

  coef_tab <- summary(model)$coefficients$cond
  fixef_cond <- fixef(model)$cond
  intercept <- if ("(Intercept)" %in% names(fixef_cond)) {
    as.numeric(fixef_cond[["(Intercept)"]])
  } else {
    0
  }
  linkinv_func <- model$family$linkinv
  if (!is.function(linkinv_func)) {
    linkinv_func <- make.link("logit")$linkinv
  }
  to_response_delta <- function(delta_eta) {
    as.numeric(linkinv_func(intercept + delta_eta) - linkinv_func(intercept))
  }

  ci_wald <- tryCatch(
    suppressMessages(confint(model, parm = "beta_", level = 0.95)),
    error = function(e) NULL
  )

  ci_boot <- NULL
  if (ci_method == "bootstrap") {
    ci_boot <- tryCatch({
      term_names <- names(fixef_cond)
      sim_vals <- simulate(model, nsim = n_boot, seed = seed)
      sim_list <- if (is.data.frame(sim_vals)) {
        as.list(sim_vals)
      } else if (is.list(sim_vals)) {
        sim_vals
      } else {
        list()
      }

      mf <- model.frame(model)
      response_name <- all.vars(formula(model)[[2]])[1]
      boot_mat <- matrix(
        NA_real_,
        nrow = length(sim_list),
        ncol = length(term_names),
        dimnames = list(NULL, term_names)
      )

      n_success <- 0L
      for (i in seq_along(sim_list)) {
        y_i <- as.numeric(sim_list[[i]])
        if (length(y_i) != nrow(mf)) {
          next
        }
        boot_df <- mf
        boot_df[[response_name]] <- y_i

        fit_i <- tryCatch(
          suppressWarnings(update(model, data = boot_df)),
          error = function(e) NULL
        )
        if (is.null(fit_i)) {
          next
        }

        fe_i <- tryCatch(fixef(fit_i)$cond, error = function(e) NULL)
        if (is.null(fe_i)) {
          next
        }

        for (term in term_names) {
          if (term %in% names(fe_i)) {
            boot_mat[i, term] <- as.numeric(fe_i[[term]])
          }
        }
        n_success <- n_success + 1L
      }

      message(
        "glmmTMB bootstrap refits succeeded: ",
        n_success,
        "/",
        length(sim_list)
      )

      if (n_success < 2L) {
        NULL
      } else {
        out <- matrix(
          NA_real_,
          nrow = length(term_names),
          ncol = 2,
          dimnames = list(term_names, c("2.5 %", "97.5 %"))
        )
        for (term in term_names) {
          vals <- boot_mat[, term]
          vals <- vals[is.finite(vals)]
          if (length(vals) > 1) {
            out[term, ] <- as.numeric(
              quantile(vals, probs = c(0.025, 0.975), na.rm = TRUE)
            )
          }
        }
        out
      }
    }, error = function(e) NULL)
  }

  get_ci <- function(term) {
    est <- as.numeric(fixef_cond[[term]])
    se <- as.numeric(coef_tab[term, "Std. Error"])
    ci_low <- est - 1.96 * se
    ci_up <- est + 1.96 * se

    if (
      !is.null(ci_boot) &&
      term %in% rownames(ci_boot) &&
      all(is.finite(ci_boot[term, ]))
    ) {
      ci_low <- as.numeric(ci_boot[term, 1])
      ci_up <- as.numeric(ci_boot[term, 2])
    } else if (!is.null(ci_wald)) {
      rn <- rownames(ci_wald)
      idx <- which(rn %in% c(paste0("cond.", term), term))
      if (length(idx) > 0) {
        ci_low <- as.numeric(ci_wald[idx[1], 1])
        ci_up <- as.numeric(ci_wald[idx[1], 2])
      }
    }

    c(ci_low, ci_up)
  }

  effect_dict <- list()
  re_cond <- tryCatch(ranef(model)$cond, error = function(e) NULL)
  re_study <- NULL
  if (!is.null(re_cond) && "SS" %in% names(re_cond)) {
    re_study <- as.data.frame(re_cond$SS)
  }

  for (term in names(fixef_cond)) {
    if (term == "(Intercept)") {
      next
    }

    est <- as.numeric(fixef_cond[[term]])
    ci <- get_ci(term)
    ci_resp <- sort(c(
      to_response_delta(ci[1]),
      to_response_delta(ci[2])
    ))
    info <- list(
      mean = to_response_delta(est),
      ci_lower_2_5 = as.numeric(ci_resp[1]),
      ci_upper_97_5 = as.numeric(ci_resp[2])
    )

    if (!is.null(re_study) && term %in% colnames(re_study)) {
      deviations <- re_study[[term]]
      deviations <- deviations[!is.na(deviations)]
      if (length(deviations) > 1) {
        lower_eta <- est + as.numeric(
          quantile(deviations, probs = re_lower / 100)
        )
        upper_eta <- est + as.numeric(
          quantile(deviations, probs = re_upper / 100)
        )
        rs_resp <- sort(c(
          to_response_delta(lower_eta),
          to_response_delta(upper_eta)
        ))
        info$random_slope_lower <- as.numeric(rs_resp[1])
        info$random_slope_upper <- as.numeric(rs_resp[2])
      }
    }

    effect_dict[[term]] <- info
  }

  write_json(effect_dict, effects_output_path, auto_unbox = TRUE)
  quit(save = "no")
}

if (mode == "extract-phi") {
  if (is.null(phi_output_path) || phi_output_path == "") {
    stop("Missing --phi-output-path for extract-phi mode.")
  }
  model <- readRDS(model_path)
  phi <- NA_real_
  disp_fixef <- tryCatch(fixef(model)$disp, error = function(e) NULL)
  if (!is.null(disp_fixef) && "(Intercept)" %in% names(disp_fixef)) {
    phi <- exp(unname(disp_fixef[["(Intercept)"]]))
  }
  write_json(list(phi = as.numeric(phi)), phi_output_path, auto_unbox = TRUE)
  quit(save = "no")
}

stop(paste("Unknown mode:", mode))
