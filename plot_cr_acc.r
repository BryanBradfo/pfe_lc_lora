# Installer ggplot2 si ce n'est pas déjà fait
install.packages("ggplot2")

# Charger ggplot2
library(ggplot2)

# Créer les données sous forme de data frame
data <- data.frame(
  model = rep(c("ViT Small LoRA on MSA avec n_heads = 8",
                "ViT Small LoRA on MLP avec n_heads = 8",
                "ViT Small LoRA on MSAxMLP avec n_heads = 8",
                "ViT Small LoRA on MSA avec n_heads = 12",
                "ViT Small LoRA on MLP avec n_heads = 12",
                "ViT Small LoRA on MSAxMLP avec n_heads = 12"), each = 2),
  version = rep(c("LoRA", "Full"), times = 6),
  accuracy = c(91.4, 96.7, 96.7, 97.2, 94.2, 98.2, 98.1, 98.9, 98.1, 98.9, 97.8, 98.9),
  compression_ratio = c(790.168, 790.168, 8215.591, 8215.591, 12648.12, 12648.12,  781.921, 781.921, 9268.205, 9268.205, 11908.411, 11908.411)
)

# Définir les couleurs pour chaque modèle
model_colors <- c(
  "ViT Small LoRA on MSA avec n_heads = 8" = "darkred",
  "ViT Small LoRA on MLP avec n_heads = 8" = "darkgreen",
  "ViT Small LoRA on MSAxMLP avec n_heads = 8" = "darkblue",
  "ViT Small LoRA on MSA avec n_heads = 12" = "red",
  "ViT Small LoRA on MLP avec n_heads = 12" = "lightgreen",
  "ViT Small LoRA on MSAxMLP avec n_heads = 12" = "lightblue"
)

# Créer le plot
ggplot(data, aes(x = accuracy, y = compression_ratio, color = model, group = model)) +
  geom_point(size = 3, aes(shape = version)) +  # Ajouter des points pour chaque version
  geom_line() +  # Ajouter des segments reliant les points
  labs(x = "Accuracy", y = "Compression Ratio", title = "Compression Ratio vs Accuracy for Different ViT Small Models") +
  scale_color_manual(values = model_colors) +  # Appliquer les couleurs définies
  theme_minimal() +
  theme(legend.position = "bottom", legend.title = element_blank())


# # Installer ggplot2 si ce n'est pas déjà fait
# install.packages("ggplot2")

# # Charger ggplot2
# library(ggplot2)

# # Créer les données sous forme de data frame
# data <- data.frame(
#   model = rep(c("ViT Small LoRA on MSA avec n_heads = 8",
#                 "ViT Small LoRA on MLP avec n_heads = 8",
#                 "ViT Small LoRA on MSAxMLP avec n_heads = 8",
#                 "ViT Small LoRA on MSA avec n_heads = 12",
#                 "ViT Small LoRA on MLP avec n_heads = 12",
#                 "ViT Small LoRA on MSAxMLP avec n_heads = 12"), each = 2),
#   version = rep(c("LoRA", "Full"), times = 6),
#   accuracy = c(91.4, 96.7, 96.7, 97.2, 94.2, 98.2, 98.1, 98.9, 98.1, 98.9, 97.8, 98.9),
#   compression_ratio = c(790.168, 790.168, 8215.591, 8215.591, 12648.12, 12648.12,  781.921, 781.921, 9268.205, 9268.205, 11908.411, 11908.411)
# )

# # Créer le plot
# ggplot(data, aes(x = accuracy, y = compression_ratio, color = model, group = model)) +
#   geom_point(size = 3, aes(shape = version)) +  # Ajouter des points pour chaque version
#   geom_line() +  # Ajouter des segments reliant les points
#   labs(x = "Accuracy", y = "Compression Ratio", title = "Compression Ratio vs Accuracy for Different ViT Small Models") +
#   theme_minimal() +
#   theme(legend.position = "bottom", legend.title = element_blank())
