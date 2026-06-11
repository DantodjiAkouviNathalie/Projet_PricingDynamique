# Docker Deployment for Olist Pricing

Cette documentation explique comment construire et exécuter le projet dans des conteneurs Docker.

## Fichiers créés

- `Dockerfile` : image Docker pour l'API et l'application Streamlit
- `docker-compose.yml` : déploiement multi-service avec API et Streamlit
- `.dockerignore` : fichiers exclus du contexte Docker

## Construction de l'image

Depuis le dossier `pricing/pricing` :

```powershell
docker build -t pricing-app .
```

## Lancement des services

```powershell
docker compose up --build
```

Cela démarre :

- API sur `http://localhost:8000`
- Streamlit sur `http://localhost:8501`

## Utilisation

### API

```powershell
curl http://localhost:8000/health
```

### Streamlit

Ouvrir dans un navigateur :

```
http://localhost:8501
```

## Commandes utiles

### Exécuter en arrière-plan

```powershell
docker compose up -d --build
```

### Arrêter les services

```powershell
docker compose down
```

### Supprimer les images générées

```powershell
docker image rm pricing-app
```

## Notes

- Le Dockerfile utilise `python:3.12-slim`.
- Toutes les dépendances sont installées depuis `requirements.txt`.
- Le service Streamlit utilise le même build pour éviter la duplication de l'environnement.
